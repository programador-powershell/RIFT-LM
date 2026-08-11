import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { extname, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import analyzeApi from "../api/analyze.mjs";
import converterApi from "../api/converter.mjs";
import geyserApi from "../api/geyser.mjs";
import modelsApi from "../api/models.mjs";
import realTestLauncher from "../api/real-test.mjs";
import resultsApi from "../api/results.mjs";
import runnerApi from "../api/runner.mjs";
import testLauncher from "../api/test.mjs";

const root = resolve(fileURLToPath(new URL("..", import.meta.url)));
const port = Number(process.env.PORT || 3000);
const mime = {
  ".html": "text/html; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".py": "text/x-python; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
};

// Mesmos handlers usados em produção pela Vercel (paridade local, zero deps).
const apiRoutes = new Map([
  ["/api/models", modelsApi],
  ["/api/results", resultsApi],
  ["/api/analyze", analyzeApi],
  ["/api/test", testLauncher],
  ["/api/real-test", realTestLauncher],
  ["/api/geyser", geyserApi],
  ["/api/runner", runnerApi],
  ["/api/converter", converterApi],
]);

async function readRequestBody(request) {
  if (request.method === "GET" || request.method === "HEAD") return undefined;
  const chunks = [];
  for await (const chunk of request) chunks.push(chunk);
  return Buffer.concat(chunks);
}

function forwardHeaders(request) {
  const headers = new Headers();
  for (const [name, value] of Object.entries(request.headers)) {
    if (typeof value === "string") headers.set(name, value);
    else if (Array.isArray(value)) for (const entry of value) headers.append(name, entry);
  }
  return headers;
}

async function respondWithHandler(handler, requestUrl, request, response) {
  const body = await readRequestBody(request);
  const result = await handler.fetch(new Request(requestUrl, {
    method: request.method,
    headers: forwardHeaders(request),
    ...(body && body.length ? { body } : {}),
  }));
  const buffer = Buffer.from(await result.arrayBuffer());
  const headers = Object.fromEntries(result.headers.entries());
  delete headers["content-length"];
  response.writeHead(result.status, headers);
  response.end(buffer);
}

createServer(async (request, response) => {
  try {
    const requestUrl = new URL(request.url, `http://127.0.0.1:${port}`);
    const pathname = decodeURIComponent(requestUrl.pathname);

    // Paridade com vercel.json: /runner.py -> /api/runner (contrato §14.3).
    if (pathname === "/runner.py") {
      requestUrl.pathname = "/api/runner";
      await respondWithHandler(runnerApi, requestUrl, request, response);
      return;
    }

    // Paridade com vercel.json: /converter.py -> /api/converter (contrato
    // §26.1 — runner local auto-contido do conversor CASCADE). O match exato
    // vem ANTES do wildcard /converter/:model* (ordering-safe como na Vercel).
    if (pathname === "/converter.py") {
      requestUrl.pathname = "/api/converter";
      await respondWithHandler(converterApi, requestUrl, request, response);
      return;
    }

    // Paridade com vercel.json: /converter/:model* -> /api/test?battery=converter
    // (contrato §26.2 — a query original, ex.: ?hf_repo=...&publish=on, é
    // preservada; a validação de hf_repo/publish fica no handler /api/test).
    const converter = pathname.match(/^\/converter\/(.+)$/i);
    if (converter) {
      requestUrl.pathname = "/api/test";
      requestUrl.searchParams.set("battery", "converter");
      requestUrl.searchParams.set("model", converter[1]);
      await respondWithHandler(testLauncher, requestUrl, request, response);
      return;
    }

    // Paridade com vercel.json: /microlm -> /api/test?battery=microlm
    // (contrato §22.3 — rota SEM segmento de modelo: o MicroLM é o próprio
    // modelo de referência; a query original, ex.: ?publish=off, é preservada).
    if (pathname === "/microlm") {
      requestUrl.pathname = "/api/test";
      requestUrl.searchParams.set("battery", "microlm");
      await respondWithHandler(testLauncher, requestUrl, request, response);
      return;
    }

    // Paridade com vercel.json: /c3/:tech/:model* -> /api/test?battery=c3&...
    // Como na Vercel, :tech aceita qualquer segmento (incl. "all" para a fila
    // serial completa) e a validação fica no handler /api/test.
    const c3 = pathname.match(/^\/c3\/([A-Za-z0-9_-]+)\/(.+)$/i);
    if (c3) {
      requestUrl.pathname = "/api/test";
      requestUrl.searchParams.set("battery", "c3");
      requestUrl.searchParams.set("technology", c3[1].toLowerCase());
      requestUrl.searchParams.set("model", c3[2]);
      await respondWithHandler(testLauncher, requestUrl, request, response);
      return;
    }

    // Paridade com vercel.json: /final/:tech/:model* -> /api/test?battery=final&...
    // (fase final C4/C5/C6, contrato §16). Como na Vercel, :tech aceita qualquer
    // segmento (incl. "all") e a validação fica no handler /api/test.
    const finalPhase = pathname.match(/^\/final\/([A-Za-z0-9_-]+)\/(.+)$/i);
    if (finalPhase) {
      requestUrl.pathname = "/api/test";
      requestUrl.searchParams.set("battery", "final");
      requestUrl.searchParams.set("technology", finalPhase[1].toLowerCase());
      requestUrl.searchParams.set("model", finalPhase[2]);
      await respondWithHandler(testLauncher, requestUrl, request, response);
      return;
    }

    // Paridade com vercel.json: /geyser/:model* -> /api/geyser?model=...
    const geyser = pathname.match(/^\/geyser\/(.+)$/i);
    if (geyser) {
      requestUrl.pathname = "/api/geyser";
      requestUrl.searchParams.set("model", geyser[1]);
      await respondWithHandler(geyserApi, requestUrl, request, response);
      return;
    }

    // Paridade com vercel.json: /gguf/:model* -> /api/test?battery=gguf&model=...
    // (contrato §11 — a query original, ex.: ?quant=UD-Q2_K_XL, é preservada).
    const gguf = pathname.match(/^\/gguf\/(.+)$/i);
    if (gguf) {
      requestUrl.pathname = "/api/test";
      requestUrl.searchParams.set("battery", "gguf");
      requestUrl.searchParams.set("model", gguf[1]);
      await respondWithHandler(testLauncher, requestUrl, request, response);
      return;
    }

    // Paridade com vercel.json: /cap/:model* -> /api/test?battery=cap&model=...
    const cap = pathname.match(/^\/cap\/(.+)$/i);
    if (cap) {
      requestUrl.pathname = "/api/test";
      requestUrl.searchParams.set("battery", "cap");
      requestUrl.searchParams.set("model", cap[1]);
      await respondWithHandler(testLauncher, requestUrl, request, response);
      return;
    }

    // Paridade com vercel.json: caminhos amigáveis apontam para /api/real-test.
    const friendly = pathname.match(/^\/(rift|cascade|aether|spectra|winner)\/(.+)$/i);
    if (friendly) {
      requestUrl.pathname = "/api/real-test";
      requestUrl.searchParams.set("technology", friendly[1].toLowerCase());
      requestUrl.searchParams.set("model", friendly[2]);
      await respondWithHandler(realTestLauncher, requestUrl, request, response);
      return;
    }

    const handler = apiRoutes.get(pathname);
    if (handler) {
      await respondWithHandler(handler, requestUrl, request, response);
      return;
    }

    // Paridade com vercel.json (contrato §24.1): "/" -> index.html — o painel
    // ÚNICO do deploy. O antigo painel resumido foi deletado e as rotas legadas
    // dele deixaram de existir (caem no 404 padrão, como na Vercel).
    const relative = pathname === "/"
      ? "index.html"
      : pathname.replace(/^\/+/, "");
    const target = resolve(root, relative);
    if (target !== root && !target.startsWith(`${root}${sep}`)) {
      response.writeHead(403).end("Forbidden");
      return;
    }
    if (!(await stat(target)).isFile()) throw new Error("not-file");
    const body = await readFile(target);
    response.writeHead(200, {
      "Content-Type": mime[extname(target).toLowerCase()] || "application/octet-stream",
      "Cache-Control": "no-store",
      "X-Content-Type-Options": "nosniff",
    });
    response.end(body);
  } catch {
    response.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" }).end("Not found");
  }
}).listen(port, "127.0.0.1", () => {
  console.log(`Dashboard local: http://127.0.0.1:${port}`);
});
