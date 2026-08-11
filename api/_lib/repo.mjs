// Resolução repo-agnóstica de owner/repo e ref (docs/C3_CONTRACTS_V1.md §14.1).
// Diretório com underscore (api/_lib/): a Vercel NÃO cria função para ele —
// este módulo é só biblioteca compartilhada pelos handlers de api/*.
//
// Cadeia única (espelhada em api/* e nos scripts Python via envs exportadas):
//   repo: GITHUB_REPO → RIFT_GITHUB_REPOSITORY →
//         VERCEL_GIT_REPO_OWNER + "/" + VERCEL_GIT_REPO_SLUG →
//         fallback legado "programador-powershell/RIFT-LM"
//   ref:  RIFT_GITHUB_BRANCH → VERCEL_GIT_COMMIT_SHA (pin preferido) → "main"

// Único lugar do diretório api/ autorizado a manter a string legada (§14.1):
// serve apenas como último fallback quando nenhuma env está configurada.
const LEGACY_REPOSITORY = "programador-powershell/RIFT-LM";
const DEFAULT_REF = "main";
const REPO_RE = /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/;
const REF_RE = /^[A-Za-z0-9._/-]+$/;
const COMMIT_SHA_RE = /^[a-f0-9]{40}$/i;

/**
 * Normaliza um valor de repositório (aceita URL https/ssh do GitHub ou
 * "owner/repo" puro). Retorna null quando o valor não é utilizável — a cadeia
 * de resolução segue para o próximo candidato em vez de derrubar o endpoint.
 */
function cleanRepository(value) {
  const repository = String(value || "")
    .trim()
    .replace(/^https?:\/\/github\.com\//i, "")
    .replace(/^git@github\.com:/i, "")
    .replace(/\.git$/i, "")
    .replace(/\/+$/, "");
  return REPO_RE.test(repository) ? repository : null;
}

function cleanRef(value) {
  const ref = String(value || "").trim();
  return ref && REF_RE.test(ref) && !ref.includes("..") ? ref : null;
}

/**
 * resolveRepo(): GITHUB_REPO → RIFT_GITHUB_REPOSITORY →
 * VERCEL_GIT_REPO_OWNER/VERCEL_GIT_REPO_SLUG → fallback legado.
 * Valores inválidos são pulados (nunca lança) — endpoint público não pode cair
 * por configuração ruim.
 */
export function resolveRepo(env = process.env) {
  for (const name of ["GITHUB_REPO", "RIFT_GITHUB_REPOSITORY"]) {
    const repository = cleanRepository(env[name]);
    if (repository) return repository;
  }
  const owner = String(env.VERCEL_GIT_REPO_OWNER || "").trim();
  const slug = String(env.VERCEL_GIT_REPO_SLUG || "").trim();
  if (owner && slug) {
    const repository = cleanRepository(`${owner}/${slug}`);
    if (repository) return repository;
  }
  return LEGACY_REPOSITORY;
}

/**
 * resolveRef(): RIFT_GITHUB_BRANCH → VERCEL_GIT_COMMIT_SHA (pin preferido no
 * deploy Vercel) → "main". Nunca lança.
 */
export function resolveRef(env = process.env) {
  const branch = cleanRef(env.RIFT_GITHUB_BRANCH);
  if (branch) return branch;
  const sha = String(env.VERCEL_GIT_COMMIT_SHA || "").trim();
  if (COMMIT_SHA_RE.test(sha)) return sha;
  return DEFAULT_REF;
}

/** Base raw.githubusercontent.com para downloads versionados de scripts. */
export function rawBaseUrl(repo = resolveRepo(), ref = resolveRef()) {
  return `https://raw.githubusercontent.com/${repo}/${ref}`;
}

export const _test = {
  COMMIT_SHA_RE,
  DEFAULT_REF,
  LEGACY_REPOSITORY,
  REPO_RE,
  cleanRef,
  cleanRepository,
};
