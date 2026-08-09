import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { extname, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
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

createServer(async (request, response) => {
  try {
    const requestUrl = new URL(request.url, `http://127.0.0.1:${port}`);
    const pathname = decodeURIComponent(requestUrl.pathname);
    const friendly = pathname.match(/^\/(rift|cascade|aether|spectra)\/(.+)$/i);
    if (pathname === "/api/test" || friendly) {
      if (friendly) {
        requestUrl.pathname = "/api/test";
        requestUrl.searchParams.set("technology", friendly[1].toLowerCase());
        requestUrl.searchParams.set("model", friendly[2]);
      }
      const result = await testLauncher.fetch(new Request(requestUrl, { method: request.method }));
      const body = Buffer.from(await result.arrayBuffer());
      response.writeHead(result.status, Object.fromEntries(result.headers.entries()));
      response.end(body);
      return;
    }
    const relative = pathname === "/" ? "index.html" : pathname.replace(/^\/+/, "");
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
