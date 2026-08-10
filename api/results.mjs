import { timingSafeEqual } from "node:crypto";

const MAX_BODY_BYTES = 1024 * 1024;
const MAX_CONTEXT_BYTES = 16 * 1024;
const MAX_RECORDS = 25000;
const GITHUB_API_VERSION = "2022-11-28";
const TECHNOLOGIES = new Set(["RIFT", "CASCADE", "AETHER", "SPECTRA", "WINNER"]);
const MODEL_ID_RE = /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/;
const COMPARISON_GROUP_RE = /^[A-Za-z0-9_.:-]{8,128}$/;

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
  const hint = `${record.battery_id || ""} ${record.spec || ""}`.toUpperCase();
  if (hint.includes("WINNER")) return "WINNER";
  if (hint.includes("SPECTRA")) return "SPECTRA";
  if (hint.includes("AETHER")) return "AETHER";
  if (hint.includes("CASCADE")) return "CASCADE";
  if (hint.includes("RIFT")) return "RIFT";
  return null;
}

function finiteMetric(value, field, index, source) {
  if (value === null || value === undefined) return;
  if (!Number.isFinite(Number(value))) {
    throw new ApiError(`${source}.${field} invalido no indice ${index}`, 400);
  }
}

function normalizeRecord(record, index, source) {
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
  if (modelId && !MODEL_ID_RE.test(modelId)) {
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
    "aether_tok_s", "spectra_tok_s", "winner_tok_s",
    "baseline_ram_bytes", "candidate_ram_bytes", "rift_ram_bytes", "cascade_ram_bytes",
    "aether_ram_bytes", "spectra_ram_bytes", "winner_ram_bytes",
    "baseline_disk_bytes", "candidate_disk_bytes", "rift_disk_bytes", "cascade_disk_bytes",
    "aether_disk_bytes", "spectra_disk_bytes", "winner_disk_bytes",
  ]) {
    finiteMetric(record[field], field, index, source);
  }

  const comparisonGroup = record.comparison_group_id == null
    ? null
    : String(record.comparison_group_id).trim();
  if (comparisonGroup && !COMPARISON_GROUP_RE.test(comparisonGroup)) {
    throw new ApiError(`${source}.comparison_group_id invalido no indice ${index}`, 400);
  }

  if (record.comparison_context != null) {
    if (typeof record.comparison_context !== "object" || Array.isArray(record.comparison_context)) {
      throw new ApiError(`${source}.comparison_context precisa ser objeto no indice ${index}`, 400);
    }
    if (Buffer.byteLength(JSON.stringify(record.comparison_context), "utf8") > MAX_CONTEXT_BYTES) {
      throw new ApiError(`${source}.comparison_context excede 16 KiB no indice ${index}`, 400);
    }
  }

  const schemaVersion = Number(record.schema_version ?? 1);
  if (!Number.isInteger(schemaVersion) || schemaVersion < 1 || schemaVersion > 1000) {
    throw new ApiError(`${source}.schema_version invalido no indice ${index}`, 400);
  }

  const normalized = {
    ...record,
    schema_version: schemaVersion,
    technology,
    ...(modelId ? { model_id: modelId } : {}),
  };
  if (comparisonGroup) normalized.comparison_group_id = comparisonGroup;
  return normalized;
}

function validateHistory(value, source) {
  if (!Array.isArray(value)) {
    throw new ApiError(`${source} precisa ser um array JSON`, 400);
  }
  if (value.length > MAX_RECORDS) {
    throw new ApiError(`${source} excede o limite de ${MAX_RECORDS} registros`, 413);
  }
  return value.map((record, index) => normalizeRecord(record, index, source));
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
  const repository = normalizeRepository(requiredEnv("RIFT_GITHUB_REPOSITORY"));
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

export default {
  async fetch(request) {
    if (request.method !== "POST") {
      return jsonResponse(
        { ok: false, error: "Metodo nao permitido" },
        405,
        { Allow: "POST" },
      );
    }

    try {
      const expectedToken = requiredEnv("RIFT_INGEST_TOKEN");
      const authorization = request.headers.get("authorization") || "";
      const receivedToken = authorization.startsWith("Bearer ")
        ? authorization.slice(7)
        : "";
      if (!secretsMatch(receivedToken, expectedToken)) {
        return jsonResponse({ ok: false, error: "Nao autorizado" }, 401);
      }

      const payload = await readRequestJson(request);
      const records = validateHistory(payload.records ?? payload, "Payload");
      const publication = await publishToGitHub(records);
      return jsonResponse({ ok: true, publication }, 200);
    } catch (error) {
      const status = error instanceof ApiError ? error.status : 500;
      const message = error instanceof ApiError ? error.message : "Erro interno";
      return jsonResponse({ ok: false, error: message }, status);
    }
  },
};

export const _test = {
  inferTechnology,
  mergeHistories,
  normalizeRecord,
  normalizeRepository,
  normalizeTargetPath,
  recordKey,
  secretsMatch,
  validateHistory,
};
