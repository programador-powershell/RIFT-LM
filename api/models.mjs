const HUGGING_FACE_MODELS_API = "https://huggingface.co/api/models";
const SEARCH_LIMIT = 12;
const UPSTREAM_LIMIT = 30;
const REQUEST_TIMEOUT_MS = 8000;

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
  if (query.length > 80) throw new ApiError("A busca pode ter no máximo 80 caracteres");
  if (!/^[\p{L}\p{N}_.\-/ ]+$/u.test(query)) {
    throw new ApiError("A busca contém caracteres não permitidos");
  }
  return query;
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
      && model.pipeline_tag === "text-generation"
      && (!model.library_name || model.library_name === "transformers")
    ))
    .sort((left, right) => {
      const leftText = left.pipeline_tag === "text-generation" ? 1 : 0;
      const rightText = right.pipeline_tag === "text-generation" ? 1 : 0;
      return rightText - leftText || right.downloads - left.downloads || left.id.localeCompare(right.id);
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
  url.searchParams.set("pipeline_tag", "text-generation");
  url.searchParams.set("sort", "downloads");
  url.searchParams.set("direction", "-1");
  url.searchParams.set("limit", String(UPSTREAM_LIMIT));
  let response;
  try {
    response = await fetch(url, {
      headers: { Accept: "application/json", "User-Agent": "rift-model-queue/1.0" },
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
  } catch (error) {
    throw new ApiError(`Falha ao consultar o Hugging Face: ${error.message}`, 502);
  }
  if (!response.ok) throw new ApiError(`Hugging Face retornou HTTP ${response.status}`, 502);
  return normalizeModelResults(await response.json());
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
      const query = normalizeSearch(new URL(request.url).searchParams.get("q"));
      const models = await searchModels(query);
      return request.method === "HEAD"
        ? new Response(null, { status: 200, headers: { "Cache-Control": "public, s-maxage=300" } })
        : jsonResponse({ ok: true, query, models });
    } catch (error) {
      return errorResponse(error);
    }
  },
};

export const _test = { normalizeModelResults, normalizeSearch };
