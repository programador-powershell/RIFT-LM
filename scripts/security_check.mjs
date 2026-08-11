#!/usr/bin/env node
// Checks estáticos de segurança (docs/C3_CONTRACTS_V1.md §5). Sai com código 1
// em qualquer falha, com relatório legível:
//
// (a) vercel.json precisa de Content-Security-Policy (sem unsafe-eval),
//     Strict-Transport-Security e X-Content-Type-Options: nosniff globais;
// (b) varredura de segredos óbvios em arquivos texto do repositório
//     (AWS AKIA, GitHub ghp_/ghs_, sk-, Google AIza, blocos PRIVATE KEY,
//     Authorization: Bearer com token literal de 32+ caracteres).
//     Exclusões documentadas: arquivos .env.example (placeholders) e este
//     próprio script (contém os padrões da varredura);
// (c) heurística de publisher Python: todo .py da raiz, de engines/**/ e de
//     batteries/ (árvore canônica do §20) que usa RIFT_INGEST_TOKEN precisa
//     mencionar "https" e "32" em até 40 linhas de distância de algum uso do
//     token (enforcement de HTTPS + token >=32);
// (d) package.json não pode declarar dependências externas.
//
// Uso: node scripts/security_check.mjs

import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const SELF = fileURLToPath(import.meta.url);

const SKIP_DIRECTORIES = new Set([".git", ".github_cache", "node_modules", ".vercel", "__pycache__"]);
const TEXT_EXTENSIONS = new Set([
  ".mjs", ".js", ".json", ".py", ".html", ".md", ".yml", ".yaml",
  ".txt", ".sh", ".css", ".toml", ".cfg", ".ini", ".ps1", ".env",
]);

const failures = [];
function fail(section, message) {
  failures.push(`(${section}) ${message}`);
}

// ---------------------------------------------------------------------------
// (a) Cabeçalhos de segurança no vercel.json
// ---------------------------------------------------------------------------

async function checkVercelHeaders() {
  let config;
  try {
    config = JSON.parse(await readFile(path.join(ROOT, "vercel.json"), "utf8"));
  } catch (error) {
    fail("a", `vercel.json ilegível: ${error.message}`);
    return;
  }
  const allHeaders = (config.headers || []).flatMap((entry) => entry.headers || []);
  const byKey = new Map(
    allHeaders.map((header) => [String(header.key).toLowerCase(), String(header.value)]),
  );

  const csp = byKey.get("content-security-policy");
  if (!csp) {
    fail("a", "vercel.json sem Content-Security-Policy");
  } else {
    if (/unsafe-eval/.test(csp)) fail("a", "Content-Security-Policy não pode conter unsafe-eval");
    if (!/default-src/.test(csp)) fail("a", "Content-Security-Policy sem default-src");
  }

  const hsts = byKey.get("strict-transport-security");
  if (!hsts) {
    fail("a", "vercel.json sem Strict-Transport-Security");
  } else if (!/max-age=\d{4,}/.test(hsts)) {
    fail("a", "Strict-Transport-Security precisa de max-age razoável");
  }

  if (byKey.get("x-content-type-options") !== "nosniff") {
    fail("a", "vercel.json sem X-Content-Type-Options: nosniff");
  }
}

// ---------------------------------------------------------------------------
// (b) Varredura de segredos em arquivos texto
// ---------------------------------------------------------------------------

// Padrões literais. O padrão Bearer exige um token literal (a classe de
// caracteres exclui { e $, então interpolações como "Bearer ${token}" ou
// f"Bearer {token}" nunca casam).
const SECRET_PATTERNS = [
  { name: "AWS Access Key ID", re: /AKIA[0-9A-Z]{16}/ },
  { name: "GitHub PAT (ghp_)", re: /ghp_[A-Za-z0-9]{36}/ },
  { name: "GitHub App token (ghs_)", re: /ghs_[A-Za-z0-9]{36}/ },
  { name: "Secret key (sk-)", re: /\bsk-[A-Za-z0-9]{20,}/ },
  { name: "Google API key (AIza)", re: /AIza[0-9A-Za-z_-]{35}/ },
  { name: "Bloco de chave privada", re: /-----BEGIN (?:RSA|EC|DSA|OPENSSH|PGP)? ?PRIVATE KEY-----/ },
  { name: "Bearer token literal (32+)", re: /Authorization["'`]?\s*[:=]\s*["'`]?Bearer\s+[A-Za-z0-9_.\-=+/]{32,}/i },
];

async function* walkTextFiles(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      if (!SKIP_DIRECTORIES.has(entry.name)) yield* walkTextFiles(fullPath);
      continue;
    }
    if (!entry.isFile()) continue;
    const extension = path.extname(entry.name).toLowerCase();
    if (!TEXT_EXTENSIONS.has(extension)) continue;
    // Exclusões documentadas: placeholders de exemplo e este próprio script.
    if (entry.name === ".env.example") continue;
    if (path.resolve(fullPath) === path.resolve(SELF)) continue;
    yield fullPath;
  }
}

async function checkSecrets() {
  for await (const filePath of walkTextFiles(ROOT)) {
    let text;
    try {
      text = await readFile(filePath, "utf8");
    } catch {
      continue;
    }
    for (const { name, re } of SECRET_PATTERNS) {
      const match = text.match(re);
      if (match) {
        const relative = path.relative(ROOT, filePath);
        fail("b", `possível segredo (${name}) em ${relative}: ${match[0].slice(0, 24)}...`);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// (c) Heurística: publishers Python impõem HTTPS + token >= 32 chars.
// Escopo (árvore canônica, contrato §20): *.py da RAIZ (legado — hoje vazia),
// engines/**/*.py (baterias por tecnologia, recursivo — inclui
// engines/winner/cpp/ se algum .py aparecer lá) e batteries/*.py (baterias
// multi-motor: c3_methodology, final_phase, capability_eval, gguf_e2e,
// compare_generations_publisher). A heurística é simples de propósito: em até
// 40 linhas de qualquer uso de RIFT_INGEST_TOKEN precisa aparecer "https"
// (case-insensitive) e o literal "32" — os enforcements reais
// (urlparse scheme == https / len(token) < 32) contêm ambos.
// ---------------------------------------------------------------------------

const PUBLISHER_WINDOW_LINES = 40;

async function listPythonFiles(directory, { recursive = false } = {}) {
  let entries;
  try {
    entries = await readdir(directory, { withFileTypes: true });
  } catch {
    return []; // diretório ausente (ex.: raiz sem .py ou árvore parcial)
  }
  const files = [];
  for (const entry of entries) {
    const fullPath = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      if (recursive && !SKIP_DIRECTORIES.has(entry.name)) {
        files.push(...(await listPythonFiles(fullPath, { recursive })));
      }
      continue;
    }
    if (entry.isFile() && entry.name.endsWith(".py")) files.push(fullPath);
  }
  return files;
}

async function checkPythonPublishers() {
  const pythonFiles = [
    ...(await listPythonFiles(ROOT)),
    ...(await listPythonFiles(path.join(ROOT, "engines"), { recursive: true })),
    ...(await listPythonFiles(path.join(ROOT, "batteries"))),
  ].sort();

  let publishersChecked = 0;
  for (const filePath of pythonFiles) {
    const name = path.relative(ROOT, filePath).split(path.sep).join("/");
    const text = await readFile(filePath, "utf8");
    const lines = text.split(/\r?\n/);
    const tokenLines = [];
    lines.forEach((line, index) => {
      if (line.includes("RIFT_INGEST_TOKEN")) tokenLines.push(index);
    });
    if (!tokenLines.length) continue; // não é publisher
    publishersChecked += 1;

    const windowIndexes = new Set();
    for (const tokenLine of tokenLines) {
      const start = Math.max(0, tokenLine - PUBLISHER_WINDOW_LINES);
      const end = Math.min(lines.length - 1, tokenLine + PUBLISHER_WINDOW_LINES);
      for (let index = start; index <= end; index += 1) windowIndexes.add(index);
    }
    const windowText = [...windowIndexes].sort((a, b) => a - b).map((i) => lines[i]).join("\n");

    if (!/https/i.test(windowText)) {
      fail("c", `${name}: uso de RIFT_INGEST_TOKEN sem enforcement de HTTPS por perto`);
    }
    if (!/32/.test(windowText)) {
      fail("c", `${name}: uso de RIFT_INGEST_TOKEN sem verificação de tamanho mínimo (32) por perto`);
    }
  }

  // Guarda contra regressão do glob: as baterias publicam em engines/**/ e
  // batteries/ — encontrar ZERO publishers significa que a varredura ficou
  // apontando para diretórios vazios (foi exatamente o que a mudança de árvore
  // do §20 causaria com o escopo antigo, restrito à raiz).
  if (publishersChecked === 0) {
    fail("c", "nenhum publisher Python encontrado (raiz + engines/** + batteries/) — glob desatualizado?");
  }
}

// ---------------------------------------------------------------------------
// (d) package.json sem dependências externas
// ---------------------------------------------------------------------------

async function checkPackageJson() {
  let manifest;
  try {
    manifest = JSON.parse(await readFile(path.join(ROOT, "package.json"), "utf8"));
  } catch (error) {
    fail("d", `package.json ilegível: ${error.message}`);
    return;
  }
  for (const field of ["dependencies", "devDependencies", "peerDependencies", "optionalDependencies"]) {
    const value = manifest[field];
    if (value && Object.keys(value).length > 0) {
      fail("d", `package.json declara ${field}: ${Object.keys(value).join(", ")} (o projeto é zero-dependência)`);
    }
  }
}

await checkVercelHeaders();
await checkSecrets();
await checkPythonPublishers();
await checkPackageJson();

if (failures.length) {
  console.error(`security_check: ${failures.length} falha(s):`);
  for (const failure of failures) console.error(`  - ${failure}`);
  process.exit(1);
}
console.log("security_check: OK (CSP/HSTS/nosniff, sem segredos óbvios, publishers com HTTPS+token>=32, zero dependências)");
