const HUGGING_FACE_MODELS_API = "https://huggingface.co/api/models";
const SEARCH_LIMIT = 12;
const UPSTREAM_LIMIT = 30;
const REQUEST_TIMEOUT_MS = 10000;
/** Limite Colab: 110 GiB de pesos no hub (safetensors/bin). */
const COLAB_MAX_BYTES = 110 * 1024 * 1024 * 1024;
/** Mesmo formato de quant aceito pelo launcher GGUF (contrato §11). */
const GGUF_QUANT_RE = /^[A-Za-z0-9_.-]{2,32}$/;

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
      // Nº de parâmetros quando o Hub já expõe metadata safetensors (aditivo;
      // normalmente ausente na busca — null indica "não informado").
      params: Number.isFinite(Number(model?.safetensors?.total))
        ? Number(model.safetensors.total)
        : null,
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

/**
 * Metadata de picker do Hub (aditivo, best-effort): likes, downloads, gated,
 * pipeline_tag e nº de parâmetros quando presente na metadata safetensors.
 * Falhas retornam null e nunca derrubam a resposta de detalhe.
 */
/** Codifica cada segmento do id preservando a barra literal (a API do Hub
 * responde 400 para org%2Fnome no caminho — barra deve permanecer '/'). */
function encodeModelPath(id) {
  return String(id).split("/").map(encodeURIComponent).join("/");
}

async function modelPickerMetadata(modelId) {
  const id = normalizeModelId(modelId);
  if (!id) return null;
  try {
    const response = await fetch(`${HUGGING_FACE_MODELS_API}/${encodeModelPath(id)}`, {
      headers: { Accept: "application/json", "User-Agent": "rift-model-queue/1.1" },
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
    if (!response.ok) return null;
    const info = await response.json();
    if (!info || typeof info !== "object" || Array.isArray(info)) return null;
    return {
      likes: Number.isFinite(Number(info.likes)) ? Number(info.likes) : null,
      downloads: Number.isFinite(Number(info.downloads)) ? Number(info.downloads) : null,
      gated: info.gated === true || info.gated === "auto" || info.gated === "manual",
      pipeline_tag: typeof info.pipeline_tag === "string" ? info.pipeline_tag : null,
      params: Number.isFinite(Number(info?.safetensors?.total))
        ? Number(info.safetensors.total)
        : null,
    };
  } catch {
    return null;
  }
}

function normalizeQuant(value) {
  const quant = String(value || "").trim();
  if (!quant) return null;
  if (!GGUF_QUANT_RE.test(quant)) {
    throw new ApiError(
      "quant inválido; use de 2 a 32 caracteres [A-Za-z0-9_.-] (ex.: UD-Q2_K_XL)",
    );
  }
  return quant;
}

/**
 * Soma pesos da árvore de arquivos do repositório. Quando `quant` é informado
 * E o repositório é GGUF (possui >=1 arquivo .gguf), SOMENTE os .gguf cujo
 * caminho contém o tag do quant entram no gate de 110 GiB e nos campos de
 * tamanho (aditivos: quant/quant_bytes/quant_files). Sem quant, sem .gguf ou
 * sem arquivo correspondente ao quant, o comportamento padrão é preservado.
 */
function summarizeWeightTree(tree, quant = null) {
  const weightRe = /\.(safetensors|bin|pt|ckpt|pth|gguf|npz|npz\.index)$/i;
  const files = [];
  for (const entry of tree) {
    if (!entry || entry.type === "directory") continue;
    const path = String(entry.path || entry.rfilename || "");
    files.push({ path, size: Number(entry.size) || 0 });
  }
  const ggufFiles = files.filter((file) => /\.gguf$/i.test(file.path));
  if (quant && ggufFiles.length) {
    const quantLower = quant.toLowerCase();
    const quantFiles = ggufFiles.filter((file) => file.path.toLowerCase().includes(quantLower));
    if (quantFiles.length) {
      const quantBytes = quantFiles.reduce((sum, file) => sum + file.size, 0);
      return {
        total: quantBytes,
        weightFiles: quantFiles.length,
        quantInfo: { quant, quant_bytes: quantBytes, quant_files: quantFiles.length },
      };
    }
  }
  let total = 0;
  let weightFiles = 0;
  for (const file of files) {
    if (weightRe.test(file.path) || /model-\d+-of-\d+/i.test(file.path)) {
      total += file.size;
      weightFiles += 1;
    }
  }
  // Se não achou pesos tipados, soma tudo (exceto README/docs)
  if (weightFiles === 0) {
    for (const file of files) {
      const path = file.path.toLowerCase();
      if (path.endsWith(".md") || path.endsWith(".txt") || path.endsWith(".json") && !path.includes("index")) continue;
      total += file.size;
    }
  }
  const quantInfo = quant && ggufFiles.length
    ? {
      quant,
      quant_bytes: 0,
      quant_files: 0,
      quant_note: `Nenhum arquivo .gguf com o tag ${quant} foi encontrado; o tamanho padrão do repositório foi aplicado.`,
    }
    : null;
  return { total, weightFiles, quantInfo };
}

/** Soma tamanhos de pesos no repositório (safetensors/bin/pt/ckpt). */
async function modelWeightBytes(modelId, quant = null) {
  const id = normalizeModelId(modelId);
  if (!id) throw new ApiError("Model ID inválido");
  const url = `https://huggingface.co/api/models/${encodeModelPath(id)}/tree/main?recursive=true`;
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
  const { total, weightFiles, quantInfo } = summarizeWeightTree(tree, quant);
  const size_gb = total / (1024 ** 3);
  const quantApplied = Boolean(quantInfo && quantInfo.quant_files > 0);
  return {
    id,
    size_bytes: total,
    size_gb: Math.round(size_gb * 100) / 100,
    weight_files: weightFiles,
    too_large: total > COLAB_MAX_BYTES,
    max_gb: 110,
    gated_or_private: false,
    ...(quantInfo ?? {}),
    note: total > COLAB_MAX_BYTES
      ? (quantApplied
        ? `Arquivos do quant ${quantInfo.quant} ~${size_gb.toFixed(1)} GiB excedem o limite de 110 GiB da bateria Colab.`
        : `Pesos ~${size_gb.toFixed(1)} GiB excedem o limite de 110 GiB da bateria Colab.`)
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
        // Consultas em paralelo: pesos (gate de 110 GiB) + metadata de picker.
        // A metadata é aditiva e best-effort; só os pesos podem falhar a rota.
        // Com &quant= e repositório GGUF, o gate considera apenas os arquivos
        // do quant (contrato §11 — o Colab só baixa os arquivos do quant).
        const quant = normalizeQuant(params.get("quant"));
        const [info, picker] = await Promise.all([
          modelWeightBytes(modelId, quant),
          modelPickerMetadata(modelId),
        ]);
        return request.method === "HEAD"
          ? new Response(null, { status: 200 })
          : jsonResponse({ ok: true, ...(picker ?? {}), ...info });
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

export const _test = {
  normalizeModelResults,
  normalizeSearch,
  normalizeModelId,
  normalizeQuant,
  modelPickerMetadata,
  summarizeWeightTree,
  COLAB_MAX_BYTES,
};
