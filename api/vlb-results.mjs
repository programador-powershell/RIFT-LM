import { timingSafeEqual } from "node:crypto";
import { resolveRepo } from "./_lib/repo.mjs";

const MAX_BODY_BYTES = 256 * 1024;
const MAX_RECORDS = 25000;
const GITHUB_API_VERSION = "2022-11-28";
const MODEL_RE = /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/;
const DEFAULT_PATH = "data/rift_test_batteries.json";

class ApiError extends Error {
  constructor(message, status = 500) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function requiredEnv(name) {
  const value = process.env[name]?.trim();
  if (!value) throw new ApiError(`Variavel de ambiente ausente: ${name}`, 500);
  return value;
}

function secretsMatch(received, expected) {
  const a = Buffer.from(received || "", "utf8");
  const b = Buffer.from(expected || "", "utf8");
  return a.length === b.length && b.length >= 32 && timingSafeEqual(a, b);
}

function normalizePath(value) {
  const path = String(value || DEFAULT_PATH).replaceAll("\\", "/").replace(/^\/+|\/+$/g, "");
  const parts = path.split("/");
  if (!path || parts.some((part) => !part || part === "." || part === "..")) {
    throw new ApiError("RIFT_GITHUB_DATA_PATH invalido", 500);
  }
  return path;
}

function normalizeRecord(input) {
  if (!input || typeof input !== "object" || Array.isArray(input)) {
    throw new ApiError("Registro VLB invalido", 400);
  }
  const modelId = String(input.model_id || input.model || "").trim();
  if (!MODEL_RE.test(modelId)) throw new ApiError("model_id VLB invalido", 400);
  const batteryId = String(input.battery_id || "VLB_AMT_STREAMING_V1").trim();
  if (!batteryId || batteryId.length > 180) throw new ApiError("battery_id VLB invalido", 400);
  const now = new Date().toISOString();
  const runId = String(input.run_id || `vlb-${Date.now()}-${modelId.replace(/[^A-Za-z0-9]+/g, "-").slice(-60)}`);
  const metrics = input.metrics && typeof input.metrics === "object" && !Array.isArray(input.metrics)
    ? input.metrics
    : {};
  const proof = metrics?.proof && typeof metrics.proof === "object" ? metrics.proof : {};
  const conversion = metrics?.vlb?.conversion_gate && typeof metrics.vlb.conversion_gate === "object"
    ? metrics.vlb.conversion_gate
    : {};
  const enginePass = proof.engine_status === "VLB_AMT_ENGINE_PROOF_PASS"
    || (proof.conversion_verified === true && proof.runtime_verified === true && proof.amt_verified === true);
  const baselineDisk = Number(input.baseline_disk_bytes);
  const candidateDisk = Number(input.candidate_disk_bytes);
  const diskReduction = Number.isFinite(baselineDisk) && baselineDisk > 0 && Number.isFinite(candidateDisk)
    ? (1 - candidateDisk / baselineDisk) * 100
    : null;
  return {
    description: "VLB/AMT streaming re-quantization + native runtime proof",
    schema_version: 1,
    timestamp_utc: String(input.timestamp_utc || now),
    run_id: runId.slice(0, 160),
    spec: "VLB_AMT_ENGINE_V1",
    technology: "VLB",
    model_id: modelId,
    battery_id: batteryId,
    benchmark_protocol: "VLB_AMT_PROOF_FIRST_V1",
    comparison_role: "primary",
    implementation: input.implementation || {
      kind: "EXPERIMENTAL_NATIVE_RUNTIME",
      scope: "streaming_requantization_vlb_runtime_amt",
      native: proof.runtime_verified === true,
      simulated: false,
    },
    eligible_for_primary_ranking: enginePass,
    status: enginePass ? "PASS" : "EXPERIMENTAL_FAIL",
    ...(Number.isFinite(baselineDisk) ? { baseline_disk_bytes: baselineDisk } : {}),
    ...(Number.isFinite(candidateDisk) ? { candidate_disk_bytes: candidateDisk } : {}),
    quality: {
      output: {
        cosine: Number.isFinite(Number(conversion.observed_worst_cosine))
          ? Number(conversion.observed_worst_cosine)
          : null,
        nrmse: Number.isFinite(Number(conversion.observed_worst_nrmse))
          ? Number(conversion.observed_worst_nrmse)
          : null,
      },
      full_local_gate_pass: enginePass,
    },
    metrics,
    gains: diskReduction == null ? {} : { disk_reduction_pct: diskReduction },
    notes: String(input.notes || "VLB is not verified until conversion, VLB runtime and AMT all pass.").slice(0, 16000),
    measurement_scope: "VLB-DIR streaming conversion + native VLB runtime + teacher-free AMT",
  };
}

async function githubRequest(url, token, options = {}) {
  const response = await fetch(url, {
    ...options,
    cache: "no-store",
    headers: {
      Accept: "application/vnd.github+json",
      Authorization: `Bearer ${token}`,
      "User-Agent": "rift-vlb-results/0.1",
      "X-GitHub-Api-Version": GITHUB_API_VERSION,
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...options.headers,
    },
  });
  const text = await response.text();
  let body = null;
  if (text) {
    try { body = JSON.parse(text); } catch { body = { message: text }; }
  }
  if (!response.ok) {
    const error = new ApiError(`GitHub API HTTP ${response.status}: ${body?.message || "erro"}`, response.status >= 500 ? 502 : response.status);
    error.githubStatus = response.status;
    throw error;
  }
  return body;
}

async function readJson(request) {
  const declared = Number(request.headers.get("content-length") || 0);
  if (declared > MAX_BODY_BYTES) throw new ApiError("Payload VLB excede 256 KiB", 413);
  const text = await request.text();
  if (Buffer.byteLength(text, "utf8") > MAX_BODY_BYTES) throw new ApiError("Payload VLB excede 256 KiB", 413);
  try { return JSON.parse(text); } catch { throw new ApiError("JSON VLB invalido", 400); }
}

function recordKey(record) {
  return [String(record.run_id || ""), String(record.technology || ""), String(record.battery_id || "")].join("\u0000");
}

async function publish(record) {
  const token = requiredEnv("RIFT_GITHUB_TOKEN");
  const repository = resolveRepo();
  const branch = process.env.RIFT_GITHUB_BRANCH?.trim() || "main";
  const path = normalizePath(process.env.RIFT_GITHUB_DATA_PATH || DEFAULT_PATH);
  const apiRoot = `https://api.github.com/repos/${repository}`;
  const contentUrl = `${apiRoot}/contents/${path.split("/").map(encodeURIComponent).join("/")}`;

  for (let attempt = 1; attempt <= 3; attempt += 1) {
    let sha = null;
    let history = [];
    try {
      const current = await githubRequest(`${contentUrl}?ref=${encodeURIComponent(branch)}`, token);
      sha = current.sha;
      history = JSON.parse(Buffer.from(current.content, "base64").toString("utf8"));
      if (!Array.isArray(history)) throw new ApiError("Historico remoto invalido", 502);
    } catch (error) {
      if (!(error instanceof ApiError) || error.githubStatus !== 404) throw error;
    }

    const map = new Map(history.map((row) => [recordKey(row), row]));
    map.set(recordKey(record), record);
    const merged = [...map.values()];
    if (merged.length > MAX_RECORDS) throw new ApiError("Historico cheio", 507);
    const body = {
      message: `data: publish VLB benchmark ${record.run_id}`,
      content: Buffer.from(`${JSON.stringify(merged, null, 2)}\n`, "utf8").toString("base64"),
      branch,
      ...(sha ? { sha } : {}),
    };
    try {
      const result = await githubRequest(contentUrl, token, { method: "PUT", body: JSON.stringify(body) });
      return { repository, branch, path, commit_sha: result.commit?.sha || null, records: merged.length };
    } catch (error) {
      if (!(error instanceof ApiError) || error.githubStatus !== 409 || attempt === 3) throw error;
    }
  }
  throw new ApiError("Falha de concorrencia ao publicar VLB", 409);
}

export default {
  async fetch(request) {
    if (request.method !== "POST") {
      return Response.json({ ok: false, error: "Metodo nao permitido" }, { status: 405, headers: { Allow: "POST" } });
    }
    try {
      const expected = requiredEnv("RIFT_INGEST_TOKEN");
      const authorization = request.headers.get("authorization") || "";
      const received = authorization.startsWith("Bearer ") ? authorization.slice(7) : "";
      if (!secretsMatch(received, expected)) {
        return Response.json({ ok: false, error: "Nao autorizado" }, { status: 401 });
      }
      const payload = await readJson(request);
      const raw = Array.isArray(payload?.records) ? payload.records[0] : payload;
      const record = normalizeRecord(raw);
      const publication = await publish(record);
      return Response.json({ ok: true, record, publication }, { status: 200, headers: { "Cache-Control": "no-store" } });
    } catch (error) {
      const status = error instanceof ApiError ? error.status : 500;
      return Response.json({ ok: false, error: error instanceof ApiError ? error.message : "Erro interno" }, { status });
    }
  },
};

export const _test = { normalizeRecord, secretsMatch, recordKey };
