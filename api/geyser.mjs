// GET /api/geyser?model=<org/modelo> — entrega geyser_launcher.py (docs/C3_CONTRACTS_V1.md §7).
// Rota amigável: /geyser/:model* (vercel.json). Diferente dos demais launchers,
// esta rota serve o PRÓPRIO arquivo Python de engines/geyser/ (árvore canônica,
// contrato §20), com TODAS as ocorrências do placeholder __GEYSER_MODEL_ID__
// substituídas pelo modelo validado.
// Contrato de uso:
//   curl -fsSL <url> -o /content/geyser_launcher.py && python /content/geyser_launcher.py
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { resolveRepo, resolveRef, rawBaseUrl } from "./_lib/repo.mjs";

const LAUNCHER_FILENAME = "geyser_launcher.py";
// Caminho no repositório (contrato §20). O .vercelignore exclui *.py do upload,
// então o arquivo NÃO está no bundle do deploy — em produção o fallback busca o
// launcher no GitHub raw, no repo/ref resolvidos (§14.1); a leitura local só
// funciona em dev. Por isso vercel.json NÃO usa functions/includeFiles.
const LAUNCHER_REPO_PATH = "engines/geyser/geyser_launcher.py";
const RAW_FETCH_TIMEOUT_MS = 8000;
const MODEL_ID_PLACEHOLDER = "__GEYSER_MODEL_ID__";
// Mesmo formato org/modelo validado por normalizeModel (api/test.mjs) e
// MODEL_ID_RE (api/results.mjs). A classe de caracteres exclui aspas, quebras
// de linha e barras extras, então a substituição no fonte Python é segura.
const MODEL_ID_RE = /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/;

class ApiError extends Error {
  constructor(message, status = 400) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function normalizeModelId(value) {
  const raw = String(value || "").trim().replace(/^\/+|\/+$/g, "");
  let modelId = raw;
  try {
    const url = new URL(raw);
    if (!["huggingface.co", "www.huggingface.co"].includes(url.hostname.toLowerCase())) {
      throw new ApiError("A URL do modelo precisa ser do huggingface.co");
    }
    const parts = url.pathname.split("/").filter(Boolean);
    modelId = parts.slice(0, 2).join("/");
  } catch (error) {
    if (error instanceof ApiError) throw error;
    // raw não é URL: segue como org/modelo puro.
  }
  if (!MODEL_ID_RE.test(modelId)) {
    throw new ApiError("Modelo inválido; use org/modelo ou uma URL do huggingface.co");
  }
  return modelId;
}

async function readLauncherSource() {
  // 1) Leitura local (dev server / clones locais). Em produção o *.py não está
  // no bundle (.vercelignore), então estes candidatos falham e caímos no raw.
  const candidates = [
    new URL("../engines/geyser/geyser_launcher.py", import.meta.url),
    join(process.cwd(), "engines", "geyser", LAUNCHER_FILENAME),
  ];
  for (const candidate of candidates) {
    try {
      return await readFile(candidate, "utf8");
    } catch {
      // Tenta o próximo caminho candidato.
    }
  }
  // 2) Fallback de produção: GitHub raw no repo/ref resolvidos (repo-agnóstico,
  // §14.1 — pin no SHA do deploy quando disponível).
  const rawUrl = `${rawBaseUrl(resolveRepo(), resolveRef())}/${LAUNCHER_REPO_PATH}`;
  try {
    const response = await fetch(rawUrl, {
      headers: { Accept: "text/plain" },
      signal: AbortSignal.timeout(RAW_FETCH_TIMEOUT_MS),
    });
    if (response.ok) return await response.text();
    throw new ApiError(
      `${LAUNCHER_FILENAME} indisponível: GitHub raw respondeu HTTP ${response.status}`,
      502,
    );
  } catch (error) {
    if (error instanceof ApiError) throw error;
    throw new ApiError(`${LAUNCHER_FILENAME} indisponível (local e GitHub raw)`, 502);
  }
}

function renderLauncher(source, modelId) {
  return source.replaceAll(MODEL_ID_PLACEHOLDER, modelId);
}

function errorResponse(error) {
  const status = error instanceof ApiError ? error.status : 500;
  return Response.json(
    { ok: false, error: error instanceof ApiError ? error.message : "Erro interno" },
    {
      status,
      headers: { "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff" },
    },
  );
}

export default {
  async fetch(request) {
    if (request.method !== "GET") {
      return new Response("Método não permitido", {
        status: 405,
        headers: {
          Allow: "GET",
          "Cache-Control": "no-store",
          "Content-Type": "text/plain; charset=utf-8",
          "X-Content-Type-Options": "nosniff",
        },
      });
    }
    try {
      const modelId = normalizeModelId(new URL(request.url).searchParams.get("model"));
      const source = await readLauncherSource();
      const body = renderLauncher(source, modelId);
      return new Response(body, {
        status: 200,
        headers: {
          "Cache-Control": "no-store",
          "Content-Disposition": `inline; filename="${LAUNCHER_FILENAME}"`,
          "Content-Type": "text/plain; charset=utf-8",
          "X-Content-Type-Options": "nosniff",
        },
      });
    } catch (error) {
      return errorResponse(error);
    }
  },
};

export const _test = {
  LAUNCHER_FILENAME,
  LAUNCHER_REPO_PATH,
  MODEL_ID_PLACEHOLDER,
  normalizeModelId,
  readLauncherSource,
  renderLauncher,
};
