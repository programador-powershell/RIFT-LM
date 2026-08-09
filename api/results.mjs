import { timingSafeEqual } from "node:crypto";

const MAX_BODY_BYTES = 1024 * 1024;
const GITHUB_API_VERSION = "2022-11-28";

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
  if (!/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(repository)) {
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

function validateHistory(value, source) {
  if (!Array.isArray(value)) {
    throw new ApiError(`${source} precisa ser um array JSON`, 400);
  }
  for (const [index, record] of value.entries()) {
    if (!record || typeof record !== "object" || Array.isArray(record)) {
      throw new ApiError(`${source} contem registro invalido no indice ${index}`, 400);
    }
    if (!String(record.run_id || "").trim() || !String(record.battery_id || "").trim()) {
      throw new ApiError(
        `${source} exige run_id e battery_id no indice ${index}`,
        400,
      );
    }
  }
  return value;
}

function mergeHistories(remote, incoming) {
  const records = new Map();
  for (const record of [...remote, ...incoming]) {
    records.set(`${record.run_id}\u0000${record.battery_id}`, record);
  }
  return [...records.values()].sort((left, right) => {
    const leftKey = `${left.timestamp_utc || ""}\u0000${left.run_id}\u0000${left.battery_id}`;
    const rightKey = `${right.timestamp_utc || ""}\u0000${right.run_id}\u0000${right.battery_id}`;
    return leftKey.localeCompare(rightKey);
  });
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
        "User-Agent": "rift-cascade-vercel-ingest/0.4",
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
  mergeHistories,
  normalizeRepository,
  normalizeTargetPath,
  secretsMatch,
  validateHistory,
};
