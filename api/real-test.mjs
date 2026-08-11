import { rawBaseUrl, resolveRef, resolveRepo } from "./_lib/repo.mjs";
import { _test as legacy } from "./test.mjs";

const RESULTS_ENDPOINT = "https://rift-lm.vercel.app/api/results";
const RUNNER_PATH = "scripts/real_benchmark_runner.py";

function normalizePositiveInt(value, fallback, min, max) {
  const parsed = Number.parseInt(String(value ?? ""), 10);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(min, Math.min(max, parsed));
}

function requestParameters(request) {
  const url = new URL(request.url);
  const technology = legacy.normalizeTechnology(url.searchParams.get("technology"));
  const model = legacy.normalizeModel(url.searchParams.get("model"));
  const targetLayer = legacy.normalizeTargetLayer(url.searchParams.get("target_layer"));
  const device = legacy.normalizeDevice(url.searchParams.get("device"));
  const publish = legacy.normalizePublish(url.searchParams.get("publish"));
  const trustRemoteCode = ["1", "true", "yes"].includes(
    String(url.searchParams.get("trust_remote_code") || "").toLowerCase(),
  );
  const iterations = normalizePositiveInt(url.searchParams.get("iterations"), 50, 10, 500);
  const warmup = normalizePositiveInt(url.searchParams.get("warmup"), 10, 1, 100);
  const winnerArch = legacy.normalizeWinnerArch(url.searchParams.get("arch"));
  return {
    technology,
    model,
    targetLayer,
    device,
    publish,
    trustRemoteCode,
    iterations,
    warmup,
    winnerArch,
    origin: url.origin,
  };
}

function buildLauncher({
  technology,
  model,
  targetLayer,
  device,
  publish,
  trustRemoteCode,
  iterations,
  warmup,
  winnerArch = null,
  origin,
  repo = resolveRepo(),
  ref = resolveRef(),
}) {
  const runnerUrl = `${rawBaseUrl(repo, ref)}/${RUNNER_PATH}`;
  const winnerArchLine = technology === "winner" && winnerArch
    ? `os.environ["RIFT_WINNER_ARCH"] = ${JSON.stringify(winnerArch)}`
    : "";
  const args = [
    "--technology", technology,
    "--model", model.modelId,
    "--target-layer", targetLayer,
    "--device", device,
    "--iterations", String(iterations),
    "--warmup", String(warmup),
    "--publish", publish,
    "--results-endpoint", RESULTS_ENDPOINT,
    "--source-ref", ref,
  ];
  if (trustRemoteCode || model.trustRemoteCode) args.push("--trust-remote-code");

  return `#!/usr/bin/env python3
# Launcher REAL_MEASUREMENT_V3 gerado por ${origin}.
# Não publica RAM estimada e não converte Linear rows/s em Tok/s.
import os
import sys
from urllib.request import Request, urlopen

RUNNER_URL = ${JSON.stringify(runnerUrl)}
ARGS = ${JSON.stringify(args)}

def import_colab_secrets():
    try:
        from google.colab import userdata
    except Exception:
        return
    for name in ("RIFT_INGEST_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(name):
            continue
        try:
            value = userdata.get(name)
        except Exception:
            value = None
        if value:
            os.environ[name] = str(value)
    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

import_colab_secrets()
os.environ.setdefault("RIFT_RESULTS_ENDPOINT", ${JSON.stringify(RESULTS_ENDPOINT)})
# Repo/ref resolvidos no servidor (contrato §14.2): os scripts Python nunca
# adivinham o repositório — leem estas envs exportadas pela célula gerada.
os.environ.setdefault("RIFT_GITHUB_REPOSITORY", ${JSON.stringify(repo)})
os.environ.setdefault("RIFT_SOURCE_REF", ${JSON.stringify(ref)})
${winnerArchLine}
print("[REAL-METRICS] baixando runner versionado:", RUNNER_URL)
request = Request(RUNNER_URL, headers={"User-Agent": "rift-real-launcher/1.0"})
source = urlopen(request, timeout=60).read()
sys.argv = [RUNNER_URL, *ARGS]
exec(compile(source, RUNNER_URL, "exec"), {"__name__": "__main__", "__file__": RUNNER_URL})
`;
}

function errorResponse(error) {
  const status = error?.status || 500;
  return Response.json(
    { ok: false, error: error?.message || "Erro interno" },
    { status, headers: { "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff" } },
  );
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
      const parameters = requestParameters(request);
      const body = buildLauncher(parameters);
      const filename = `real-${parameters.technology}-${parameters.model.modelId.replace("/", "-")}.py`;
      return new Response(request.method === "HEAD" ? null : body, {
        status: 200,
        headers: {
          "Cache-Control": "public, max-age=0, s-maxage=60",
          "Content-Disposition": `inline; filename="${filename}"`,
          "Content-Type": "text/x-python; charset=utf-8",
          "X-Content-Type-Options": "nosniff",
        },
      });
    } catch (error) {
      return errorResponse(error);
    }
  },
};

export const _test = {
  buildLauncher,
  normalizePositiveInt,
  requestParameters,
  resolveRef,
  resolveRepo,
};
