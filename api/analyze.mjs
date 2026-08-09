import { createHash } from "node:crypto";

const GEMINI_MODEL = "gemini-2.5-flash";
const GEMINI_URL = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent`;
const MAX_BODY_BYTES = 96 * 1024;
const MAX_MODELS = 20;
const CACHE_TTL_MS = 10 * 60 * 1000;
const RATE_WINDOW_MS = 10 * 60 * 1000;
const RATE_LIMIT = 10;
const responseCache = new Map();
const requestWindows = new Map();

class ApiError extends Error {
  constructor(message, status = 500) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function jsonResponse(body, status = 200) {
  return Response.json(body, {
    status,
    headers: {
      "Cache-Control": "private, no-store",
      "X-Content-Type-Options": "nosniff",
    },
  });
}

function requiredEnv(name) {
  const value = process.env[name]?.trim();
  if (!value) throw new ApiError(`Variavel de ambiente ausente: ${name}`, 500);
  return value;
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function finiteOrNull(value, minimum = -Infinity, maximum = Infinity) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? clamp(number, minimum, maximum) : null;
}

function booleanOrNull(value) {
  return typeof value === "boolean" ? value : null;
}

function safeLabel(value, maximum) {
  return String(value || "").replace(/[^A-Za-z0-9_.:+ -]/g, "").slice(0, maximum);
}

function normalizeTechnology(value, name) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  return {
    technology: name,
    battery_id: safeLabel(value.battery_id, 120),
    status: safeLabel(value.status, 40),
    output_cosine: finiteOrNull(value.output_cosine, -1, 1),
    output_nrmse: finiteOrNull(value.output_nrmse, 0, 100),
    quality_gate_pass: booleanOrNull(value.quality_gate_pass),
    disk_reduction_pct: finiteOrNull(value.disk_reduction_pct, -10000, 100),
    ram_reduction_pct: finiteOrNull(value.ram_reduction_pct, -10000, 100),
    operation_speedup_x: finiteOrNull(value.operation_speedup_x, 0, 10000),
    stage_activation_rate: finiteOrNull(value.stage_activation_rate, 0, 1),
  };
}

function validatePayload(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new ApiError("Payload precisa ser um objeto JSON", 400);
  }
  if (!Array.isArray(value.models) || !value.models.length) {
    throw new ApiError("Payload exige models[]", 400);
  }
  if (value.models.length > MAX_MODELS) {
    throw new ApiError(`Maximo de ${MAX_MODELS} modelos por analise`, 400);
  }
  const seen = new Set();
  return value.models.map((entry, index) => {
    const modelId = String(entry?.model_id || "").trim();
    if (!/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(modelId)) {
      throw new ApiError(`model_id invalido no indice ${index}`, 400);
    }
    if (seen.has(modelId)) throw new ApiError(`model_id duplicado: ${modelId}`, 400);
    seen.add(modelId);
    const technologies = entry?.technologies || {};
    const rift = normalizeTechnology(technologies.RIFT, "RIFT");
    const cascade = normalizeTechnology(technologies.CASCADE, "CASCADE");
    const aether = normalizeTechnology(technologies.AETHER, "AETHER");
    if (!rift && !cascade && !aether) {
      throw new ApiError(`Modelo sem bateria principal: ${modelId}`, 400);
    }
    return { model_id: modelId, technologies: { RIFT: rift, CASCADE: cascade, AETHER: aether } };
  });
}

const SCORE_WEIGHTS = {
  output_cosine: 25,
  output_nrmse: 15,
  disk_reduction_pct: 20,
  ram_reduction_pct: 15,
  operation_speedup_x: 20,
  quality_gate_pass: 5,
};

function normalizedMetric(name, value) {
  if (value === null || value === undefined) return null;
  if (name === "output_cosine") return clamp((value + 1) * 50, 0, 100);
  if (name === "output_nrmse") return 100 * (1 - clamp(value / 0.1, 0, 1));
  if (name === "disk_reduction_pct" || name === "ram_reduction_pct") {
    return clamp(value, 0, 100);
  }
  if (name === "operation_speedup_x") return clamp(value, 0, 1) * 100;
  if (name === "quality_gate_pass") return value === true ? 100 : value === false ? 0 : null;
  return null;
}

function scoreTechnology(modelId, technology) {
  let weighted = 0;
  let availableWeight = 0;
  const components = {};
  for (const [name, weight] of Object.entries(SCORE_WEIGHTS)) {
    const score = normalizedMetric(name, technology[name]);
    components[name] = score;
    if (score !== null) {
      weighted += score * weight;
      availableWeight += weight;
    }
  }
  const coverage = availableWeight / 100;
  const rawScore = availableWeight ? weighted / availableWeight : 0;
  const score = rawScore * (0.65 + 0.35 * coverage);
  return {
    model_id: modelId,
    technology: technology.technology,
    score: Number(score.toFixed(2)),
    raw_score: Number(rawScore.toFixed(2)),
    coverage_pct: Number((coverage * 100).toFixed(0)),
    quality_gate_pass: technology.quality_gate_pass,
    status: technology.status,
    components,
  };
}

function buildRanking(models) {
  const ranking = [];
  for (const model of models) {
    for (const technology of [
      model.technologies.RIFT,
      model.technologies.CASCADE,
      model.technologies.AETHER,
    ]) {
      if (technology) ranking.push(scoreTechnology(model.model_id, technology));
    }
  }
  ranking.sort((left, right) =>
    right.score - left.score ||
    right.coverage_pct - left.coverage_pct ||
    left.model_id.localeCompare(right.model_id) ||
    left.technology.localeCompare(right.technology),
  );
  return ranking.map((entry, index) => ({ position: index + 1, ...entry }));
}

const ANALYSIS_SCHEMA = {
  type: "object",
  additionalProperties: false,
  properties: {
    global_summary: { type: "string", description: "Resumo geral em português do Brasil, máximo 300 caracteres." },
    analyses: {
      type: "array",
      maxItems: MAX_MODELS,
      items: {
        type: "object",
        additionalProperties: false,
        properties: {
          model_id: { type: "string" },
          recommendation: { type: "string", enum: ["RIFT", "CASCADE", "AETHER", "INCONCLUSIVO"] },
          confidence: { type: "integer", minimum: 0, maximum: 100 },
          summary: { type: "string", description: "Veredito em português do Brasil, máximo 300 caracteres." },
          decisive_metrics: { type: "array", maxItems: 3, items: { type: "string" } },
          caveats: { type: "array", maxItems: 3, items: { type: "string" } },
        },
        required: ["model_id", "recommendation", "confidence", "summary", "decisive_metrics", "caveats"],
      },
    },
  },
  required: ["global_summary", "analyses"],
};

function buildPrompt(models, ranking) {
  const compactRanking = ranking.map(({ model_id, technology, score, coverage_pct }) => ({
    model_id,
    technology,
    score,
    coverage_pct,
  }));
  return [
    "Voce e um analista de benchmarks de compressao e execucao de LLMs.",
    "Analise SOMENTE os dados JSON fornecidos; nao use reputacao externa do modelo.",
    "Para cada model_id, recomende RIFT, CASCADE, AETHER ou INCONCLUSIVO.",
    "Compare somente tecnologias presentes. Se houver menos de duas, responda INCONCLUSIVO.",
    "Nunca recomende uma tecnologia sem bateria principal para o modelo.",
    "Priorize quality gate, output cosine/NRMSE, depois disco, RAM e speedup da operacao.",
    "Latencia de Linear nao e Tok/s. Prefetch simulado e kernel nao nativo devem aparecer nas ressalvas quando aplicavel.",
    "A bateria AETHER atual usa base ternaria 2-bit, mas HQR-ANS, P-IO e SRFA ainda nao sao implementacoes nativas.",
    "O ranking composto e uma referencia deterministica, nao substitui a interpretacao das metricas.",
    "Responda em portugues do Brasil no schema solicitado.",
    JSON.stringify({ models, deterministic_ranking: compactRanking }),
  ].join("\n");
}

function buildGeminiBody(prompt, legacy = false) {
  const responseFormat = { text: { mimeType: "application/json", schema: ANALYSIS_SCHEMA } };
  return {
    contents: [{ role: "user", parts: [{ text: prompt }] }],
    generationConfig: legacy
      ? { temperature: 0.15, maxOutputTokens: 4096, responseMimeType: "application/json", responseSchema: ANALYSIS_SCHEMA }
      : { temperature: 0.15, maxOutputTokens: 4096, responseFormat },
  };
}

async function callGemini(apiKey, prompt) {
  let lastError;
  for (const legacy of [false, true]) {
    let response;
    try {
      response = await fetch(GEMINI_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json", "x-goog-api-key": apiKey },
        body: JSON.stringify(buildGeminiBody(prompt, legacy)),
        signal: AbortSignal.timeout(45000),
      });
    } catch (error) {
      throw new ApiError(`Falha ao conectar com Gemini: ${error.message}`, 502);
    }
    const text = await response.text();
    let body;
    try { body = text ? JSON.parse(text) : null; } catch { body = null; }
    if (!response.ok) {
      lastError = new ApiError(`Gemini retornou HTTP ${response.status}: ${body?.error?.message || "erro desconhecido"}`, 502);
      if (response.status === 400 && !legacy) continue;
      throw lastError;
    }
    const output = body?.candidates?.[0]?.content?.parts
      ?.map((part) => part.text || "")
      .join("")
      .trim();
    if (!output) throw new ApiError("Gemini nao retornou texto analisavel", 502);
    try { return JSON.parse(output); } catch { throw new ApiError("Gemini retornou JSON invalido", 502); }
  }
  throw lastError || new ApiError("Gemini indisponivel", 502);
}

function validateGeminiAnalysis(value, models) {
  if (!value || !Array.isArray(value.analyses)) throw new ApiError("Analise estruturada invalida", 502);
  const byModel = new Map(value.analyses.map((analysis) => [String(analysis.model_id), analysis]));
  const analyses = models.map((model) => {
    const source = byModel.get(model.model_id);
    if (!source) throw new ApiError(`Gemini omitiu o modelo ${model.model_id}`, 502);
    let recommendation = ["RIFT", "CASCADE", "AETHER", "INCONCLUSIVO"].includes(source.recommendation)
      ? source.recommendation
      : "INCONCLUSIVO";
    const available = ["RIFT", "CASCADE", "AETHER"].filter(
      (technology) => Boolean(model.technologies[technology]),
    );
    if (available.length < 2 || (recommendation !== "INCONCLUSIVO" && !available.includes(recommendation))) {
      recommendation = "INCONCLUSIVO";
    }
    return {
      model_id: model.model_id,
      recommendation,
      confidence: Math.round(finiteOrNull(source.confidence, 0, 100) ?? 0),
      summary: String(source.summary || "Analise sem resumo.").slice(0, 400),
      decisive_metrics: Array.isArray(source.decisive_metrics)
        ? source.decisive_metrics.slice(0, 3).map((item) => String(item).slice(0, 180)) : [],
      caveats: Array.isArray(source.caveats)
        ? source.caveats.slice(0, 3).map((item) => String(item).slice(0, 180)) : [],
    };
  });
  return { global_summary: String(value.global_summary || "").slice(0, 500), analyses };
}

async function readRequestJson(request) {
  const declared = Number(request.headers.get("content-length") || 0);
  if (declared > MAX_BODY_BYTES) throw new ApiError("Payload excede 96 KiB", 413);
  const text = await request.text();
  if (Buffer.byteLength(text, "utf8") > MAX_BODY_BYTES) throw new ApiError("Payload excede 96 KiB", 413);
  try { return JSON.parse(text); } catch { throw new ApiError("Corpo nao e JSON valido", 400); }
}

function enforceSameOrigin(request) {
  const origin = request.headers.get("origin");
  if (!origin) throw new ApiError("Origin obrigatorio", 403);
  let originUrl;
  try { originUrl = new URL(origin); } catch { throw new ApiError("Origin invalido", 403); }
  const requestUrl = new URL(request.url);
  if (originUrl.host !== requestUrl.host || originUrl.protocol !== requestUrl.protocol) {
    throw new ApiError("Origin nao autorizado", 403);
  }
}

function enforceRateLimit(request) {
  const key = (request.headers.get("x-forwarded-for") || "unknown").split(",")[0].trim();
  const now = Date.now();
  const recent = (requestWindows.get(key) || []).filter((timestamp) => now - timestamp < RATE_WINDOW_MS);
  if (recent.length >= RATE_LIMIT) throw new ApiError("Limite temporario de analises excedido", 429);
  recent.push(now);
  requestWindows.set(key, recent);
}

function cacheKey(models) {
  return createHash("sha256").update(JSON.stringify(models)).digest("hex");
}

export default {
  async fetch(request) {
    if (request.method !== "POST") return jsonResponse({ ok: false, error: "Metodo nao permitido" }, 405);
    try {
      enforceSameOrigin(request);
      enforceRateLimit(request);
      const models = validatePayload(await readRequestJson(request));
      const ranking = buildRanking(models);
      const key = cacheKey(models);
      const cached = responseCache.get(key);
      if (cached && Date.now() - cached.createdAt < CACHE_TTL_MS) {
        return jsonResponse({ ok: true, cached: true, model: GEMINI_MODEL, ranking, ...cached.analysis });
      }
      const apiKey = requiredEnv("API_GOOGLE");
      const gemini = await callGemini(apiKey, buildPrompt(models, ranking));
      const analysis = validateGeminiAnalysis(gemini, models);
      responseCache.set(key, { createdAt: Date.now(), analysis });
      if (responseCache.size > 20) responseCache.delete(responseCache.keys().next().value);
      return jsonResponse({ ok: true, cached: false, model: GEMINI_MODEL, ranking, ...analysis });
    } catch (error) {
      const status = error instanceof ApiError ? error.status : 500;
      return jsonResponse({ ok: false, error: error instanceof ApiError ? error.message : "Erro interno" }, status);
    }
  },
};

export const _test = {
  buildGeminiBody,
  buildPrompt,
  buildRanking,
  enforceSameOrigin,
  normalizeTechnology,
  scoreTechnology,
  validateGeminiAnalysis,
  validatePayload,
};
