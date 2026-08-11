import { createHash, timingSafeEqual } from "node:crypto";
import { readFile } from "node:fs/promises";
import { resolveRepo } from "./_lib/repo.mjs";

const MAX_BODY_BYTES = 1024 * 1024;
const MAX_CONTEXT_BYTES = 16 * 1024;
const MAX_RECORDS = 25000;
const GITHUB_API_VERSION = "2022-11-28";
// GEYSER (§7), CAP (§9, bateria de capacidades por modelo), GGUF (§11,
// B0_GGUF_RUNTIME_SETUP / P1_GGUF_E2E_TOKS — NUNCA elegível na política do
// winner) e MICROLM (§22, 7ª tecnologia tipo MODELO — também NUNCA elegível)
// entraram em 2026-08-10.
const TECHNOLOGIES = new Set(["RIFT", "CASCADE", "AETHER", "SPECTRA", "WINNER", "GEYSER", "CAP", "GGUF", "MICROLM"]);
const MODEL_ID_RE = /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/;
// model_id de registro aceita sufixo opcional ":<tag>" (contrato §11: registros
// CAP sobre GGUF usam "org/modelo-GGUF:UD-Q2_K_XL"). Repositórios GitHub
// continuam validados pelo MODEL_ID_RE estrito.
const RECORD_MODEL_ID_RE = /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+(:[A-Za-z0-9_.-]{1,64})?$/;
const COMPARISON_GROUP_RE = /^[A-Za-z0-9_.:-]{8,128}$/;

// GET /api/results (público, dados já publicados no GitHub). O repositório é
// resolvido pela cadeia única do contrato §14.1 (api/_lib/repo.mjs) — o
// fallback legado vive apenas lá.
const HISTORY_DEFAULT_PATH = "data/rift_test_batteries.json";
const HISTORY_FETCH_TIMEOUT_MS = 8000;
const HISTORY_CACHE_CONTROL = "public, max-age=0, s-maxage=15, stale-while-revalidate=60";

// Whitelist explícita de campos de topo aceitos em cada registro publicado.
// Cobre data/record-schema-example.json + campos do schema v2. Chaves fora da
// lista (e fora dos aliases rift_*/cascade_*/aether_*/spectra_*/winner_*/
// geyser_*/gguf_*/microlm_*) são descartadas; a resposta do POST informa
// quantas foram removidas.
const RECORD_TOP_LEVEL_FIELDS = new Set([
  "description",
  "schema_version",
  "timestamp_utc",
  "run_id",
  "spec",
  "technology",
  "model_id",
  "battery_id",
  "benchmark_protocol",
  "comparison_role",
  "comparison_group_id",
  "comparison_context",
  "implementation",
  "eligible_for_primary_ranking",
  "status",
  "baseline_tok_s",
  "candidate_tok_s",
  "baseline_ram_bytes",
  "candidate_ram_bytes",
  "baseline_disk_bytes",
  "candidate_disk_bytes",
  "quality",
  "metrics",
  "gains",
  "notes",
  "measurement_scope",
]);
const RECORD_ALIAS_PREFIX_RE = /^(rift|cascade|aether|spectra|winner|geyser|gguf|microlm)_/i;
const CAPPED_OBJECT_FIELDS = ["comparison_context", "implementation", "quality", "metrics", "gains"];
const CAPPED_STRING_FIELDS = ["notes", "measurement_scope", "description", "spec"];

// Política do WINNER dinâmico (docs/C3_CONTRACTS_V1.md §1) — espelhada em Python.
// CAP NUNCA é elegível (§9): registros de capacidade avaliam o modelo baseline,
// não uma tecnologia de otimização. GGUF NUNCA é elegível (§11 — nota do enum):
// é uma série de runtime/artefato, não uma tecnologia candidata ao winner.
// MICROLM NUNCA é elegível (§22.1): é um MODELO de referência, não um
// otimizador — como CAP, fica fora desta lista PERMANENTEMENTE.
const WINNER_ELIGIBLE_TECHS = ["RIFT", "AETHER", "CASCADE", "SPECTRA", "GEYSER"];
const WINNER_TIE_ORDER = ["CASCADE", "RIFT", "AETHER", "SPECTRA", "GEYSER"];
// SCORE_WEIGHTS_V2 (contrato §25) — objetivo "computador convencional"
// (4 núcleos, 8 GB RAM livre, sem GPU): RAM 30 > disco 10 por construção.
// Qualidade 40 (cosine 25 + nrmse 10 + gate 5) • RAM 30 • velocidade 20 •
// disco 10. Normalizações do §1 inalteradas; espelhado em api/analyze.mjs,
// index.html e winner_m0 Python.
const SCORE_WEIGHTS = { cosine: 25, nrmse: 10, disk: 10, ram: 30, speedup: 20, gate: 5 };

class ApiError extends Error {
  constructor(message, status = 500) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function jsonResponse(body, status = 200, extraHeaders = {}) {
  return Response.json(body, {
    status,
    headers: {
      "Cache-Control": "no-store",
      "X-Content-Type-Options": "nosniff",
      ...extraHeaders,
    },
  });
}

function requiredEnv(name) {
  const value = process.env[name]?.trim();
  if (!value) {
    throw new ApiError(`Variavel de ambiente ausente: ${name}`, 500);
  }
  return value;
}

function secretsMatch(received, expected) {
  const receivedBuffer = Buffer.from(received || "", "utf8");
  const expectedBuffer = Buffer.from(expected || "", "utf8");
  return (
    receivedBuffer.length === expectedBuffer.length &&
    expectedBuffer.length >= 32 &&
    timingSafeEqual(receivedBuffer, expectedBuffer)
  );
}

function normalizeRepository(value) {
  const repository = value
    .trim()
    .replace(/^https?:\/\/github\.com\//i, "")
    .replace(/^git@github\.com:/i, "")
    .replace(/\.git$/i, "")
    .replace(/\/$/, "");
  if (!MODEL_ID_RE.test(repository)) {
    throw new ApiError("RIFT_GITHUB_REPOSITORY invalido", 500);
  }
  return repository;
}

function normalizeTargetPath(value) {
  const targetPath = value.replaceAll("\\", "/").replace(/^\/+|\/+$/g, "");
  const parts = targetPath.split("/");
  if (!targetPath || parts.some((part) => !part || part === "." || part === "..")) {
    throw new ApiError("RIFT_GITHUB_DATA_PATH invalido", 500);
  }
  return targetPath;
}

function inferTechnology(record) {
  const explicit = String(record.technology || "").trim().toUpperCase();
  if (explicit) return TECHNOLOGIES.has(explicit) ? explicit : null;
  const batteryId = String(record.battery_id || "").trim().toUpperCase();
  // CAP_* é prefixo reservado da bateria de capacidades (§8 nível 4 / §9).
  if (batteryId.startsWith("CAP_")) return "CAP";
  // MicroLM (§22): B0_MICROLM_*/P1_MICROLM_* são prefixos reservados da
  // bateria MICROLM_M0_V1 — 7ª tecnologia, tipo MODELO (nunca winner).
  if (batteryId.startsWith("B0_MICROLM_") || batteryId.startsWith("P1_MICROLM_")) return "MICROLM";
  const hint = `${record.battery_id || ""} ${record.spec || ""}`.toUpperCase();
  if (hint.includes("WINNER")) return "WINNER";
  if (hint.includes("GEYSER")) return "GEYSER";
  if (hint.includes("SPECTRA")) return "SPECTRA";
  if (hint.includes("AETHER")) return "AETHER";
  if (hint.includes("CASCADE")) return "CASCADE";
  if (hint.includes("RIFT")) return "RIFT";
  // GGUF (§11) por último: P1_GGUF_<TECH>_CODEC_TENSOR pertence à tecnologia
  // do codec (capturada acima); só B0_GGUF_*/P1_GGUF_E2E_* caem aqui.
  if (hint.includes("GGUF")) return "GGUF";
  return null;
}

function finiteMetric(value, field, index, source) {
  if (value === null || value === undefined) return;
  if (!Number.isFinite(Number(value))) {
    throw new ApiError(`${source}.${field} invalido no indice ${index}`, 400);
  }
}

function whitelistRecordFields(record) {
  const kept = {};
  let dropped = 0;
  for (const [key, value] of Object.entries(record)) {
    if (RECORD_TOP_LEVEL_FIELDS.has(key) || RECORD_ALIAS_PREFIX_RE.test(key)) {
      kept[key] = value;
    } else {
      dropped += 1;
    }
  }
  return { kept, dropped };
}

function enforceFieldBudgets(record, index, source) {
  for (const field of CAPPED_OBJECT_FIELDS) {
    const value = record[field];
    if (value == null) continue;
    if (typeof value !== "object" || Array.isArray(value)) {
      throw new ApiError(`${source}.${field} precisa ser objeto no indice ${index}`, 400);
    }
    if (Buffer.byteLength(JSON.stringify(value), "utf8") > MAX_CONTEXT_BYTES) {
      throw new ApiError(`${source}.${field} excede 16 KiB no indice ${index}`, 400);
    }
  }
  for (const field of CAPPED_STRING_FIELDS) {
    const value = record[field];
    if (value == null) continue;
    if (Buffer.byteLength(String(value), "utf8") > MAX_CONTEXT_BYTES) {
      throw new ApiError(`${source}.${field} excede 16 KiB no indice ${index}`, 400);
    }
  }
}

function normalizeRecord(record, index, source, stats) {
  if (!record || typeof record !== "object" || Array.isArray(record)) {
    throw new ApiError(`${source} contem registro invalido no indice ${index}`, 400);
  }

  const runId = String(record.run_id || "").trim();
  const batteryId = String(record.battery_id || "").trim();
  if (!runId || !batteryId) {
    throw new ApiError(`${source} exige run_id e battery_id no indice ${index}`, 400);
  }
  if (runId.length > 160 || batteryId.length > 180) {
    throw new ApiError(`${source} possui identificador excessivamente longo no indice ${index}`, 400);
  }

  const modelId = String(record.model_id || record.model || "").trim();
  if (modelId && !RECORD_MODEL_ID_RE.test(modelId)) {
    throw new ApiError(`${source}.model_id invalido no indice ${index}`, 400);
  }

  const technology = inferTechnology(record);
  if (!technology) {
    throw new ApiError(`${source}.technology invalida/ausente no indice ${index}`, 400);
  }

  if (record.timestamp_utc) {
    const timestamp = Date.parse(String(record.timestamp_utc));
    if (!Number.isFinite(timestamp)) {
      throw new ApiError(`${source}.timestamp_utc invalido no indice ${index}`, 400);
    }
  }

  for (const field of [
    "baseline_tok_s", "candidate_tok_s", "rift_tok_s", "cascade_tok_s",
    "aether_tok_s", "spectra_tok_s", "winner_tok_s", "geyser_tok_s", "gguf_tok_s", "microlm_tok_s",
    "baseline_ram_bytes", "candidate_ram_bytes", "rift_ram_bytes", "cascade_ram_bytes",
    "aether_ram_bytes", "spectra_ram_bytes", "winner_ram_bytes", "geyser_ram_bytes", "gguf_ram_bytes", "microlm_ram_bytes",
    "baseline_disk_bytes", "candidate_disk_bytes", "rift_disk_bytes", "cascade_disk_bytes",
    "aether_disk_bytes", "spectra_disk_bytes", "winner_disk_bytes", "geyser_disk_bytes", "gguf_disk_bytes", "microlm_disk_bytes",
  ]) {
    finiteMetric(record[field], field, index, source);
  }

  const comparisonGroup = record.comparison_group_id == null
    ? null
    : String(record.comparison_group_id).trim();
  if (comparisonGroup && !COMPARISON_GROUP_RE.test(comparisonGroup)) {
    throw new ApiError(`${source}.comparison_group_id invalido no indice ${index}`, 400);
  }

  enforceFieldBudgets(record, index, source);

  const schemaVersion = Number(record.schema_version ?? 1);
  if (!Number.isInteger(schemaVersion) || schemaVersion < 1 || schemaVersion > 1000) {
    throw new ApiError(`${source}.schema_version invalido no indice ${index}`, 400);
  }

  // Sem spread arbitrário: apenas campos da whitelist entram no histórico.
  const { kept, dropped } = whitelistRecordFields(record);
  if (stats && dropped) stats.droppedKeys += dropped;

  const normalized = {
    ...kept,
    schema_version: schemaVersion,
    technology,
    ...(modelId ? { model_id: modelId } : {}),
  };
  if (comparisonGroup) normalized.comparison_group_id = comparisonGroup;
  return normalized;
}

function validateHistory(value, source, stats) {
  if (!Array.isArray(value)) {
    throw new ApiError(`${source} precisa ser um array JSON`, 400);
  }
  if (value.length > MAX_RECORDS) {
    throw new ApiError(`${source} excede o limite de ${MAX_RECORDS} registros`, 413);
  }
  return value.map((record, index) => normalizeRecord(record, index, source, stats));
}

function recordKey(record) {
  return [
    String(record.run_id || ""),
    String(record.technology || inferTechnology(record) || ""),
    String(record.battery_id || ""),
  ].join("\u0000");
}

function mergeHistories(remote, incoming) {
  // Append-only by run. A repeated upload of the exact same run/battery is an
  // idempotent upsert, but a new run never deletes the previous one. The UI is
  // free to show only the newest snapshot while the raw history stays auditable.
  const records = new Map();
  for (const record of remote) records.set(recordKey(record), record);
  for (const record of incoming) records.set(recordKey(record), record);

  const merged = [...records.values()].sort((left, right) => {
    const leftKey = `${left.timestamp_utc || ""}\u0000${left.run_id}\u0000${left.technology || ""}\u0000${left.battery_id}`;
    const rightKey = `${right.timestamp_utc || ""}\u0000${right.run_id}\u0000${right.technology || ""}\u0000${right.battery_id}`;
    return leftKey.localeCompare(rightKey);
  });
  if (merged.length > MAX_RECORDS) {
    throw new ApiError(
      `Historico atingiu ${merged.length} registros; arquive execucoes antigas antes de publicar`,
      507,
    );
  }
  return merged;
}

async function githubRequest(url, token, options = {}) {
  let response;
  try {
    response = await fetch(url, {
      ...options,
      cache: "no-store",
      headers: {
        Accept: "application/vnd.github+json",
        Authorization: `Bearer ${token}`,
        "User-Agent": "rift-cascade-aether-spectra-winner-vercel-ingest/0.9",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
        ...(options.body ? { "Content-Type": "application/json" } : {}),
        ...options.headers,
      },
    });
  } catch (error) {
    throw new ApiError(`Falha ao conectar com a API do GitHub: ${error.message}`, 502);
  }

  const text = await response.text();
  let body = null;
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = { message: text };
    }
  }
  if (!response.ok) {
    const error = new ApiError(
      `GitHub API retornou HTTP ${response.status}: ${body?.message || "erro desconhecido"}`,
      response.status === 401 || response.status === 403 ? 502 : response.status,
    );
    error.githubStatus = response.status;
    throw error;
  }
  return body;
}

async function readRequestJson(request) {
  const declaredLength = Number(request.headers.get("content-length") || 0);
  if (declaredLength > MAX_BODY_BYTES) {
    throw new ApiError("Payload excede 1 MiB", 413);
  }
  const text = await request.text();
  if (Buffer.byteLength(text, "utf8") > MAX_BODY_BYTES) {
    throw new ApiError("Payload excede 1 MiB", 413);
  }
  try {
    return JSON.parse(text);
  } catch {
    throw new ApiError("Corpo da requisicao nao e um JSON valido", 400);
  }
}

async function publishToGitHub(incoming) {
  const token = requiredEnv("RIFT_GITHUB_TOKEN");
  // Repo de publicação: env explícita (validada com erro alto — escrita no
  // repositório errado é inaceitável) ou a cadeia resolveRepo() do §14.1.
  const configuredRepository = process.env.GITHUB_REPO?.trim()
    || process.env.RIFT_GITHUB_REPOSITORY?.trim();
  const repository = configuredRepository
    ? normalizeRepository(configuredRepository)
    : resolveRepo();
  const configuredBranch = process.env.RIFT_GITHUB_BRANCH?.trim();
  const targetPath = normalizeTargetPath(
    process.env.RIFT_GITHUB_DATA_PATH || "data/rift_test_batteries.json",
  );
  const apiRoot = `https://api.github.com/repos/${repository}`;
  const repositoryInfo = await githubRequest(apiRoot, token);
  const branch = configuredBranch || repositoryInfo.default_branch || "main";
  if (!/^[A-Za-z0-9._/-]+$/.test(branch) || branch.includes("..")) {
    throw new ApiError("RIFT_GITHUB_BRANCH invalido", 500);
  }

  const contentUrl = `${apiRoot}/contents/${targetPath
    .split("/")
    .map(encodeURIComponent)
    .join("/")}`;

  for (let attempt = 1; attempt <= 3; attempt += 1) {
    let sha;
    let remote = [];
    try {
      const current = await githubRequest(
        `${contentUrl}?ref=${encodeURIComponent(branch)}`,
        token,
      );
      sha = current.sha;
      if (current.encoding !== "base64" || typeof current.content !== "string") {
        throw new ApiError("Conteudo remoto nao retornou no formato base64", 502);
      }
      const decoded = Buffer.from(current.content, "base64").toString("utf8");
      remote = validateHistory(JSON.parse(decoded), "Historico remoto");
    } catch (error) {
      if (!(error instanceof ApiError) || error.githubStatus !== 404) {
        if (error instanceof SyntaxError) {
          throw new ApiError("Historico remoto nao e um JSON valido", 502);
        }
        throw error;
      }
    }

    const merged = mergeHistories(remote, incoming);
    const runIds = [...new Set(incoming.map((record) => String(record.run_id)))];
    const runLabel = runIds.slice(0, 2).join(", ") || "sem-run-id";
    const update = {
      message: `data: publish benchmark results ${runLabel}`,
      content: Buffer.from(`${JSON.stringify(merged, null, 2)}\n`, "utf8").toString(
        "base64",
      ),
      branch,
      ...(sha ? { sha } : {}),
    };

    try {
      const result = await githubRequest(contentUrl, token, {
        method: "PUT",
        body: JSON.stringify(update),
      });
      return {
        repository,
        branch,
        path: targetPath,
        records: merged.length,
        incoming_records: incoming.length,
        history_mode: "append-only",
        commit_sha: result.commit?.sha || null,
        commit_url: result.commit?.html_url || null,
      };
    } catch (error) {
      if (!(error instanceof ApiError) || error.githubStatus !== 409 || attempt === 3) {
        throw error;
      }
    }
  }
  throw new ApiError("Nao foi possivel publicar depois de tres tentativas", 409);
}

// ---------------------------------------------------------------------------
// Política de seleção da arquitetura do WINNER (docs/C3_CONTRACTS_V1.md §1).
// Função pura, espelhada por select_winner_architecture(records) em Python.
// ---------------------------------------------------------------------------

function clampNumber(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function finiteOrNull(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function winnerReductionPct(record, kind) {
  const direct = finiteOrNull(record?.gains?.[`${kind}_reduction_pct`]);
  if (direct !== null) return direct;
  const technology = String(record?.technology || "").trim().toLowerCase();
  const baseline = finiteOrNull(record?.[`baseline_${kind}_bytes`]);
  const candidate = finiteOrNull(
    record?.[`candidate_${kind}_bytes`] ?? record?.[`${technology}_${kind}_bytes`],
  );
  if (baseline === null || candidate === null || baseline <= 0) return null;
  return (1 - candidate / baseline) * 100;
}

function winnerScoreComponents(record) {
  const cosine = finiteOrNull(record?.quality?.output?.cosine);
  const nrmse = finiteOrNull(record?.quality?.output?.nrmse);
  const disk = winnerReductionPct(record, "disk");
  const ram = winnerReductionPct(record, "ram");
  const speedup = finiteOrNull(record?.metrics?.operation?.speedup_x);
  const gate = record?.quality?.full_local_gate_pass;
  return {
    cosine: cosine === null ? null : clampNumber((cosine + 1) * 50, 0, 100),
    nrmse: nrmse === null ? null : 100 * (1 - clampNumber(nrmse / 0.1, 0, 1)),
    disk: disk === null ? null : clampNumber(disk, 0, 100),
    ram: ram === null ? null : clampNumber(ram, 0, 100),
    speedup: speedup === null ? null : clampNumber(speedup, 0, 1) * 100,
    gate: gate === true ? 100 : gate === false ? 0 : null,
  };
}

function winnerRecordScore(record) {
  const components = winnerScoreComponents(record);
  let weighted = 0;
  let availableWeight = 0;
  for (const [name, weight] of Object.entries(SCORE_WEIGHTS)) {
    const value = components[name];
    if (value === null) continue;
    weighted += value * weight;
    availableWeight += weight;
  }
  if (!availableWeight) return null;
  // Normalização canônica do desempate (contrato §1 "Normalização do desempate"):
  // média ponderada dos presentes × (0.65 + 0.35 * coverage) — mesmas fórmulas
  // de normalizedMetric/scoreTechnology em api/analyze.mjs e _record_score em Python.
  const coverage = availableWeight / 100;
  return (weighted / availableWeight) * (0.65 + 0.35 * coverage);
}

function winnerEligiblePrimaryRecords(records) {
  const eligible = [];
  for (const record of Array.isArray(records) ? records : []) {
    if (!record || typeof record !== "object" || Array.isArray(record)) continue;
    const technology = String(record.technology || "").trim().toUpperCase();
    // Somente as tecnologias elegíveis (§1) contam; CAP e WINNER ficam fora.
    if (!WINNER_ELIGIBLE_TECHS.includes(technology)) continue;
    if (String(record.comparison_role || "") !== "primary") continue;
    const status = String(record.status || "").trim().toUpperCase();
    if (status !== "PASS" && status !== "EXPERIMENTAL_PASS") continue;
    if (record.quality && record.quality.full_local_gate_pass === false) continue;
    const modelId = String(record.model_id || "").trim();
    if (!modelId || modelId.startsWith("synthetic/")) continue;
    eligible.push({ record, technology, modelId });
  }
  return eligible;
}

export function selectWinnerArchitecture(records) {
  const eligible = winnerEligiblePrimaryRecords(records);
  const modelsByTechnology = new Map(
    WINNER_TIE_ORDER.map((technology) => [technology, new Set()]),
  );
  const latestByTechnologyModelBattery = new Map();
  for (const { record, technology, modelId } of eligible) {
    modelsByTechnology.get(technology).add(modelId);
    const key = [technology, modelId, String(record.battery_id || "")].join(" ");
    const timestamp = Date.parse(String(record.timestamp_utc || "")) || 0;
    const previous = latestByTechnologyModelBattery.get(key);
    if (!previous || timestamp >= previous.timestamp) {
      latestByTechnologyModelBattery.set(key, { timestamp, record, technology });
    }
  }

  // A iteração segue WINNER_TIE_ORDER: em empate persistente a primeira da
  // ordem (o incumbente CASCADE) permanece selecionada.
  let best = null;
  for (const technology of WINNER_TIE_ORDER) {
    const modelCount = modelsByTechnology.get(technology).size;
    const scores = [...latestByTechnologyModelBattery.values()]
      .filter((entry) => entry.technology === technology)
      .map((entry) => winnerRecordScore(entry.record))
      .filter((score) => score !== null);
    const averageScore = scores.length
      ? scores.reduce((total, score) => total + score, 0) / scores.length
      : 0;
    const beatsBest =
      !best ||
      modelCount > best.modelCount ||
      (modelCount === best.modelCount && averageScore > best.averageScore);
    if (beatsBest) best = { technology, modelCount, averageScore };
  }

  // Sem dados (ou empate total): a iteração em WINNER_TIE_ORDER garante
  // que o incumbente CASCADE permanece selecionado.
  return best ? best.technology : "CASCADE";
}

// ---------------------------------------------------------------------------
// GET /api/results — histórico público com fallback para o bundle do deploy.
// ---------------------------------------------------------------------------

function historyRepository() {
  // Cadeia única do contrato §14.1 (GITHUB_REPO → RIFT_GITHUB_REPOSITORY →
  // VERCEL_GIT_REPO_OWNER/SLUG → fallback legado). resolveRepo() nunca lança:
  // configuração inválida não pode derrubar o endpoint público.
  return resolveRepo();
}

function historyBranch() {
  const branch = process.env.RIFT_GITHUB_BRANCH?.trim() || "main";
  return /^[A-Za-z0-9._/-]+$/.test(branch) && !branch.includes("..") ? branch : "main";
}

function historyDataPath() {
  try {
    return normalizeTargetPath(process.env.RIFT_GITHUB_DATA_PATH || HISTORY_DEFAULT_PATH);
  } catch {
    return HISTORY_DEFAULT_PATH;
  }
}

async function loadPublishedHistory() {
  const targetPath = historyDataPath();
  const rawUrl = `https://raw.githubusercontent.com/${historyRepository()}/${historyBranch()}/${targetPath
    .split("/")
    .map(encodeURIComponent)
    .join("/")}`;

  try {
    const response = await fetch(rawUrl, {
      cache: "no-store",
      signal: AbortSignal.timeout(HISTORY_FETCH_TIMEOUT_MS),
      headers: {
        Accept: "application/json",
        "User-Agent": "rift-cascade-aether-spectra-winner-results-get/1.0",
      },
    });
    if (!response.ok) throw new Error(`GitHub raw HTTP ${response.status}`);
    const parsed = JSON.parse(await response.text());
    if (!Array.isArray(parsed)) throw new Error("historico remoto nao e um array");
    return { records: parsed, source: "github_raw" };
  } catch {
    // Fallback: arquivo estático empacotado no deploy.
  }

  try {
    const bundled = JSON.parse(
      await readFile(new URL(`../${targetPath}`, import.meta.url), "utf8"),
    );
    if (!Array.isArray(bundled)) throw new Error("historico local nao e um array");
    return { records: bundled, source: "deploy_bundle" };
  } catch {
    throw new ApiError("Historico indisponivel no GitHub raw e no bundle do deploy", 502);
  }
}

async function handleGetResults(request) {
  const { records, source } = await loadPublishedHistory();
  const recordsJson = JSON.stringify(records);
  const etag = `"sha256-${createHash("sha256").update(recordsJson).digest("hex").slice(0, 32)}"`;
  const headers = {
    "Cache-Control": HISTORY_CACHE_CONTROL,
    ETag: etag,
    "X-Content-Type-Options": "nosniff",
    "X-History-Source": source,
  };

  const ifNoneMatch = request.headers.get("if-none-match") || "";
  const matches = ifNoneMatch
    .split(",")
    .map((candidate) => candidate.trim().replace(/^W\//, ""));
  if (matches.includes(etag) || matches.includes("*")) {
    return new Response(null, { status: 304, headers });
  }

  const body = {
    generated_at: new Date().toISOString(),
    count: records.length,
    records,
    source,
  };
  if (request.method === "HEAD") {
    return new Response(null, {
      status: 200,
      headers: { ...headers, "Content-Type": "application/json; charset=utf-8" },
    });
  }
  return Response.json(body, { status: 200, headers });
}

export default {
  async fetch(request) {
    try {
      if (request.method === "GET" || request.method === "HEAD") {
        // Público por contrato (§4): o histórico já é publicado em repositório
        // aberto; o GET apenas o serve com cache curto e ETag.
        return await handleGetResults(request);
      }
      if (request.method !== "POST") {
        return jsonResponse(
          { ok: false, error: "Metodo nao permitido" },
          405,
          { Allow: "GET, HEAD, POST" },
        );
      }

      const expectedToken = requiredEnv("RIFT_INGEST_TOKEN");
      const authorization = request.headers.get("authorization") || "";
      const receivedToken = authorization.startsWith("Bearer ")
        ? authorization.slice(7)
        : "";
      if (!secretsMatch(receivedToken, expectedToken)) {
        return jsonResponse({ ok: false, error: "Nao autorizado" }, 401);
      }

      const payload = await readRequestJson(request);
      const stats = { droppedKeys: 0 };
      const records = validateHistory(payload.records ?? payload, "Payload", stats);
      const publication = await publishToGitHub(records);
      return jsonResponse(
        { ok: true, publication, dropped_unknown_keys: stats.droppedKeys },
        200,
      );
    } catch (error) {
      const status = error instanceof ApiError ? error.status : 500;
      const message = error instanceof ApiError ? error.message : "Erro interno";
      return jsonResponse({ ok: false, error: message }, status);
    }
  },
};

export const _test = {
  RECORD_ALIAS_PREFIX_RE,
  RECORD_MODEL_ID_RE,
  TECHNOLOGIES,
  WINNER_ELIGIBLE_TECHS,
  WINNER_TIE_ORDER,
  inferTechnology,
  historyBranch,
  historyDataPath,
  historyRepository,
  mergeHistories,
  normalizeRecord,
  normalizeRepository,
  normalizeTargetPath,
  recordKey,
  secretsMatch,
  selectWinnerArchitecture,
  validateHistory,
  whitelistRecordFields,
  winnerRecordScore,
  winnerScoreComponents,
};
