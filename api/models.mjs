const HUGGING_FACE_MODELS_API = "https://huggingface.co/api/models";
const SEARCH_LIMIT = 12;
const UPSTREAM_LIMIT = 30;
const REQUEST_TIMEOUT_MS = 10000;
/** Limite Colab: 110 GiB de pesos no hub (safetensors/bin). */
const COLAB_MAX_BYTES = 110 * 1024 * 1024 * 1024;

class ApiError extends Error {
  constructor(message, status = 400) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function normalizeSearch(value) {
  const query = String(value || "").trim().replace(/\s+/g, " ");
  if (query.length < 2) throw new ApiError("Digite pelo menos 2 caracteres");
  if (query.length > 120) throw new ApiError("A busca pode ter no máximo 120 caracteres");
  if (!/^[\p{L}\p{N}_.\-/ ]+$/u.test(query)) {
    throw new ApiError("A busca contém caracteres não permitidos");
  }
  return query;
}

function normalizeModelId(value) {
  const raw = String(value || "").trim();
  try {
    const url = new URL(raw);
    if (!["huggingface.co", "www.huggingface.co", "hf.co"].includes(url.hostname.toLowerCase())) {
      throw new Error();
    }
    const parts = url.pathname.split("/").filter(Boolean);
    if (parts.length < 2) throw new Error();
    return `${parts[0]}/${parts[1]}`;
  } catch {
    if (/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(raw)) return raw;
    return null;
  }
}

function normalizeModelResults(value) {
  if (!Array.isArray(value)) throw new ApiError("Resposta inválida do Hugging Face", 502);
  return value
    .filter((model) => model && typeof model === "object" && !model.private)
    .map((model) => ({
      id: String(model.id || model.modelId || "").trim(),
      downloads: Number.isFinite(Number(model.downloads)) ? Number(model.downloads) : 0,
      likes: Number.isFinite(Number(model.likes)) ? Number(model.likes) : 0,
      pipeline_tag: typeof model.pipeline_tag === "string" ? model.pipeline_tag : null,
      library_name: typeof model.library_name === "string" ? model.library_name : null,
      gated: model.gated === true || model.gated === "auto" || model.gated === "manual",
    }))
    .filter((model) => (
      /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(model.id)
      && !model.id.toLowerCase().includes("internal-testing")
    ))
    .sort((left, right) => {
      const score = (m) => {
        let s = 0;
        if (m.pipeline_tag === "text-generation" || m.pipeline_tag === "any-to-any") s += 2;
        if (!m.library_name || m.library_name === "transformers") s += 1;
        return s;
      };
      return score(right) - score(left) || right.downloads - left.downloads || left.id.localeCompare(right.id);
    })
    .slice(0, SEARCH_LIMIT);
}

function jsonResponse(body, status = 200) {
  return Response.json(body, {
    status,
    headers: {
      "Cache-Control": status === 200
        ? "public, max-age=0, s-maxage=300, stale-while-revalidate=86400"
        : "no-store",
      "X-Content-Type-Options": "nosniff",
    },
  });
}

function errorResponse(error) {
  if (error instanceof ApiError) return jsonResponse({ ok: false, error: error.message }, error.status);
  return jsonResponse({ ok: false, error: "Erro interno" }, 500);
}

async function searchModels(query) {
  const url = new URL(HUGGING_FACE_MODELS_API);
  url.searchParams.set("search", query);
  url.searchParams.set("sort", "downloads");
  url.searchParams.set("direction", "-1");
  url.searchParams.set("limit", String(UPSTREAM_LIMIT));
  let response;
  try {
    response = await fetch(url, {
      headers: { Accept: "application/json", "User-Agent": "rift-model-queue/1.1" },
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
  } catch (error) {
    throw new ApiError(`Falha ao consultar o Hugging Face: ${error.message}`, 502);
  }
  if (!response.ok) throw new ApiError(`Hugging Face retornou HTTP ${response.status}`, 502);
  return normalizeModelResults(await response.json());
}

/** Soma tamanhos de pesos no repositório (safetensors/bin/pt/ckpt). */
async function modelWeightBytes(modelId) {
  const id = normalizeModelId(modelId);
  if (!id) throw new ApiError("Model ID inválido");
  const url = `https://huggingface.co/api/models/${encodeURIComponent(id)}/tree/main?recursive=true`;
  let response;
  try {
    response = await fetch(url, {
      headers: { Accept: "application/json", "User-Agent": "rift-model-queue/1.1" },
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
  } catch (error) {
    throw new ApiError(`Falha ao consultar tamanho no Hugging Face: ${error.message}`, 502);
  }
  if (response.status === 401 || response.status === 403) {
    return {
      id,
      size_bytes: null,
      size_gb: null,
      too_large: false,
      gated_or_private: true,
      note: "Repositório gated/privado — tamanho não consultável sem token; a bateria pode falhar sem HF_TOKEN.",
    };
  }
  if (response.status === 404) throw new ApiError(`Modelo não encontrado: ${id}`, 404);
  if (!response.ok) throw new ApiError(`Hugging Face retornou HTTP ${response.status}`, 502);
  const tree = await response.json();
  if (!Array.isArray(tree)) throw new ApiError("Árvore de arquivos inválida do Hugging Face", 502);
  const weightRe = /\.(safetensors|bin|pt|ckpt|pth|gguf|npz|npz\.index)$/i;
  let total = 0;
  let weightFiles = 0;
  for (const entry of tree) {
    if (!entry || entry.type === "directory") continue;
    const path = String(entry.path || entry.rfilename || "");
    const size = Number(entry.size) || 0;
    if (weightRe.test(path) || /model-\d+-of-\d+/i.test(path)) {
      total += size;
      weightFiles += 1;
    }
  }
  // Se não achou pesos tipados, soma tudo (exceto README/docs)
  if (weightFiles === 0) {
    for (const entry of tree) {
      if (!entry || entry.type === "directory") continue;
      const path = String(entry.path || entry.rfilename || "").toLowerCase();
      if (path.endsWith(".md") || path.endsWith(".txt") || path.endsWith(".json") && !path.includes("index")) continue;
      total += Number(entry.size) || 0;
    }
  }
  const size_gb = total / (1024 ** 3);
  return {
    id,
    size_bytes: total,
    size_gb: Math.round(size_gb * 100) / 100,
    weight_files: weightFiles,
    too_large: total > COLAB_MAX_BYTES,
    max_gb: 110,
    gated_or_private: false,
    note: total > COLAB_MAX_BYTES
      ? `Pesos ~${size_gb.toFixed(1)} GiB excedem o limite de 110 GiB da bateria Colab.`
      : null,
  };
}

export default {
  async fetch(request) {
    if (!["GET", "HEAD"].includes(request.method)) {
      return new Response("Método não permitido", {
        status: 405,
        headers: { Allow: "GET, HEAD", "Content-Type": "text/plain; charset=utf-8" },
      });
    }
    try {
      const params = new URL(request.url).searchParams;
      const modelId = params.get("id");
      if (modelId) {
        const info = await modelWeightBytes(modelId);
        return request.method === "HEAD"
          ? new Response(null, { status: 200 })
          : jsonResponse({ ok: true, ...info });
      }
      const query = normalizeSearch(params.get("q"));
      const models = await searchModels(query);
      return request.method === "HEAD"
        ? new Response(null, { status: 200, headers: { "Cache-Control": "public, s-maxage=300" } })
        : jsonResponse({ ok: true, query, models });
    } catch (error) {
      return errorResponse(error);
    }
  },
};

export const _test = { normalizeModelResults, normalizeSearch, normalizeModelId, COLAB_MAX_BYTES };
