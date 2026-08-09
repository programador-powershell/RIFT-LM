const REPOSITORY = "programador-powershell/RIFT-LM";
const DEFAULT_REF = "main";
const RESULTS_ENDPOINT = "https://rift-lm.vercel.app/api/results";
const TOKENIZER_DEPENDENCIES = {
  sentencepiece: "sentencepiece>=0.2.0",
  tiktoken: "tiktoken>=0.7.0",
};

const TECHNOLOGIES = {
  rift: {
    label: "RIFT",
    script: "rift_m0_phase1_test_v035_auto_batteries.py",
    arguments: ["--mode", "phase1"],
  },
  cascade: {
    label: "CASCADE",
    script: "cascade_m0_phase1_test_v030_auto_batteries.py",
    arguments: [],
  },
  aether: {
    label: "AETHER",
    script: "aether_m0_phase1_test_v100_auto_batteries.py",
    arguments: ["--mode", "phase1"],
  },
  spectra: {
    label: "SPECTRA",
    script: "SPECTRA_Colab_Test_M0.py",
    arguments: ["--mode", "phase1"],
  },
  winner: {
    label: "WINNER",
    script: "winner_m0_phase1_test_v080_auto_batteries.py",
    arguments: ["--mode", "phase1"],
  },
};

const MODEL_ALIASES = {
  "kimi-k3": {
    modelId: "moonshotai/Kimi-K3",
    trustRemoteCode: true,
    warning: "Kimi-K3 exige Transformers 4.56.2 e infraestrutura distribuída; o benchmark Colab de dispositivo único não consegue carregá-lo.",
    compatibility: {
      colabSupported: false,
      transformersVersion: "4.56.2",
      totalParameters: 2_800_000_000_000,
      minimumPackedWeightBytes: 1_400_000_000_000,
      reason: (
        "Kimi-K3 tem 2,8 trilhões de parâmetros. Mesmo no limite teórico de 4 bits, "
        + "somente os pesos exigiriam pelo menos 1,4 TB; esta bateria carrega o modelo "
        + "completo em um único dispositivo. Use infraestrutura distribuída compatível "
        + "com vLLM/SGLang/TokenSpeed ou escolha um checkpoint que caiba no Colab."
      ),
    },
  },
};

class ApiError extends Error {
  constructor(message, status = 400) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function normalizeTechnology(value) {
  const key = String(value || "").trim().toLowerCase();
  if (!Object.hasOwn(TECHNOLOGIES, key)) {
    throw new ApiError("Tecnologia inválida; use rift, cascade, aether, spectra ou winner");
  }
  return key;
}

function normalizeModel(value) {
  const raw = String(value || "").trim().replace(/^\/+|\/+$/g, "");
  const alias = MODEL_ALIASES[raw.toLowerCase()];
  if (alias) return { ...alias, alias: raw };
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
  }
  const knownModel = Object.values(MODEL_ALIASES).find(
    (entry) => entry.modelId.toLowerCase() === modelId.toLowerCase(),
  );
  if (knownModel) return { ...knownModel, alias: null };
  if (!/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(modelId)) {
    throw new ApiError(
      "Modelo inválido; use org/modelo, uma URL do Hugging Face ou o alias Kimi-K3",
    );
  }
  const lowerModelId = modelId.toLowerCase();
  if (lowerModelId.includes("kimi-k3")) {
    return { ...MODEL_ALIASES["kimi-k3"], modelId, alias: null };
  }
  if (lowerModelId.includes("deepseek-v4")) {
    return {
      modelId,
      trustRemoteCode: false,
      warning: null,
      compatibility: {
        colabSupported: false,
        totalParameters: 284_000_000_000,
        minimumPackedWeightBytes: 142_000_000_000,
        reason: (
          "DeepSeek V4 Flash possui cerca de 284 bilhões de parâmetros totais. "
          + "Mesmo no limite teórico de 4 bits, somente os pesos exigem cerca de "
          + "142 GB, sem contar KV cache e buffers; a bateria Colab atual carrega "
          + "o checkpoint completo em um único dispositivo."
        ),
      },
      alias: null,
    };
  }
  if (lowerModelId.includes("gguf")) {
    return {
      modelId,
      trustRemoteCode: false,
      warning: null,
      compatibility: {
        colabSupported: false,
        reason: (
          "As baterias Python analisam camadas do checkpoint Transformers original. "
          + "Um repositório GGUF é destinado a runtimes como llama.cpp e não pode ser "
          + "carregado por esta bateria como se fosse um checkpoint Safetensors. "
          + "Selecione o model ID original sem o sufixo -GGUF."
        ),
      },
      alias: null,
    };
  }
  if (lowerModelId.includes("nvfp4")) {
    return {
      modelId,
      trustRemoteCode: false,
      warning: null,
      compatibility: {
        colabSupported: false,
        reason: (
          "Este checkpoint usa NVFP4/MTP e requer um loader e kernels específicos "
          + "para hardware NVIDIA compatível. A bateria atual usa AutoModel do "
          + "Transformers e não pode tratá-lo como um checkpoint FP16/BF16 comum."
        ),
      },
      alias: null,
    };
  }
  return {
    modelId,
    trustRemoteCode: false,
    warning: null,
    compatibility: { colabSupported: true },
    alias: null,
  };
}

function normalizeTargetLayer(value) {
  const target = String(value || "auto").trim() || "auto";
  if (!/^(auto|[A-Za-z0-9_.-]+)$/.test(target)) {
    throw new ApiError("target_layer inválido");
  }
  return target;
}

function normalizeDevice(value) {
  const device = String(value || "cuda").trim().toLowerCase();
  if (!["cpu", "cuda"].includes(device)) throw new ApiError("device precisa ser cpu ou cuda");
  return device;
}

function normalizePublish(value) {
  const mode = String(value || "required").trim().toLowerCase();
  if (!["auto", "required", "off"].includes(mode)) {
    throw new ApiError("publish precisa ser auto, required ou off");
  }
  return mode;
}

function deploymentRef() {
  const sha = String(process.env.VERCEL_GIT_COMMIT_SHA || "").trim();
  return /^[a-f0-9]{40}$/i.test(sha) ? sha : DEFAULT_REF;
}

function buildLauncher({
  technology,
  model,
  origin,
  targetLayer = "auto",
  device = "cuda",
  publish = "required",
  trustRemoteCode = false,
  ref = deploymentRef(),
}) {
  const definition = TECHNOLOGIES[technology];
  const scriptUrl = `https://raw.githubusercontent.com/${REPOSITORY}/${ref}/${definition.script}`;
  const argumentsList = [
    ...definition.arguments,
    "--model", model.modelId,
    "--target-layer", targetLayer,
    "--device", device,
    "--publish", publish,
  ];
  if (trustRemoteCode || model.trustRemoteCode) argumentsList.push("--trust-remote-code");
  const warnings = [
    model.warning,
    model.trustRemoteCode || trustRemoteCode
      ? "trust_remote_code está habilitado: o repositório do modelo poderá executar código no Colab."
      : null,
  ].filter(Boolean);
  const compatibility = model.compatibility || { colabSupported: true };
  return `#!/usr/bin/env python3
# Launcher gerado por ${origin} para ${definition.label}.
# O benchmark usa GPU do Colab; a Vercel apenas entrega este inicializador.
import os
import json
import shutil
import subprocess
import sys
from urllib.request import Request, urlopen

SCRIPT_URL = ${JSON.stringify(scriptUrl)}
MODEL_ID = ${JSON.stringify(model.modelId)}
ARGS = ${JSON.stringify(argumentsList)}
WARNINGS = ${JSON.stringify(warnings)}
COMPATIBILITY = json.loads(r'''${JSON.stringify(compatibility)}''')
PUBLISH_MODE = ${JSON.stringify(publish)}
TOKENIZER_DEPENDENCIES = json.loads(r'''${JSON.stringify(TOKENIZER_DEPENDENCIES)}''')

def format_bytes(value):
    units = ("B", "KB", "MB", "GB", "TB", "PB")
    amount = float(value)
    for unit in units:
        if amount < 1000 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1000

def available_gpu_memory_bytes():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        values = [int(line.strip()) for line in output.splitlines() if line.strip()]
        return max(values, default=0) * 1024 * 1024
    except (FileNotFoundError, ValueError, subprocess.SubprocessError):
        return 0

def enforce_compatibility():
    if COMPATIBILITY.get("colabSupported", True):
        return
    minimum = int(COMPATIBILITY.get("minimumPackedWeightBytes") or 0)
    content_root = "/content" if os.path.isdir("/content") else "."
    free_disk = shutil.disk_usage(content_root).free
    gpu_memory = available_gpu_memory_bytes()
    required_transformers = COMPATIBILITY.get("transformersVersion")
    print("[LAUNCHER] BLOQUEADO: este modelo não é compatível com a bateria Colab atual.")
    print("[LAUNCHER] Motivo:", COMPATIBILITY.get("reason", "Recursos insuficientes."))
    if minimum:
        print("[LAUNCHER] Limite teórico dos pesos:", format_bytes(minimum))
        print("[LAUNCHER] Disco livre detectado:", format_bytes(free_disk))
        print("[LAUNCHER] Maior GPU detectada:", format_bytes(gpu_memory))
    if required_transformers:
        print("[LAUNCHER] Compatibilidade do código remoto: transformers==" + required_transformers)
    print("[LAUNCHER] Nenhum peso foi baixado e nenhum resultado foi publicado.")
    raise SystemExit(2)

def enforce_publish_settings():
    if PUBLISH_MODE == "off":
        return
    token = os.environ.get("RIFT_INGEST_TOKEN", "").strip()
    if not token:
        print("[LAUNCHER] ERRO: RIFT_INGEST_TOKEN não chegou ao subprocesso.")
        print("[LAUNCHER] Copie a célula completa gerada pelo dashboard; ela lê o Secret no kernel do Colab e o transfere com segurança.")
        raise SystemExit(2)
    if len(token) < 32:
        print("[LAUNCHER] ERRO: RIFT_INGEST_TOKEN precisa ter pelo menos 32 caracteres.")
        raise SystemExit(2)

def ensure_tokenizer_dependencies():
    missing = []
    for module, package in TOKENIZER_DEPENDENCIES.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if not missing:
        print("[LAUNCHER] Tokenizadores opcionais prontos: sentencepiece + tiktoken")
        return
    print("[LAUNCHER] Instalando dependências de tokenizer:", ", ".join(missing))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
    for module in TOKENIZER_DEPENDENCIES:
        __import__(module)

os.environ.setdefault("RIFT_RESULTS_ENDPOINT", ${JSON.stringify(RESULTS_ENDPOINT)})
os.environ.setdefault("RIFT_SOURCE_REF", ${JSON.stringify(ref)})
print("[LAUNCHER] Tecnologia: ${definition.label} | Modelo:", MODEL_ID)
for warning in WARNINGS:
    print("[LAUNCHER] AVISO:", warning)
enforce_compatibility()
enforce_publish_settings()
ensure_tokenizer_dependencies()
print("[LAUNCHER] Baixando bateria versionada:", SCRIPT_URL)
request = Request(SCRIPT_URL, headers={"User-Agent": "rift-test-launcher/1.0"})
source = urlopen(request, timeout=60).read()
sys.argv = [SCRIPT_URL, *ARGS]
exec(compile(source, SCRIPT_URL, "exec"), {"__name__": "__main__", "__file__": SCRIPT_URL})
`;
}

function requestParameters(request) {
  const url = new URL(request.url);
  const technology = normalizeTechnology(url.searchParams.get("technology"));
  const model = normalizeModel(url.searchParams.get("model"));
  const targetLayer = normalizeTargetLayer(url.searchParams.get("target_layer"));
  const device = normalizeDevice(url.searchParams.get("device"));
  const publish = normalizePublish(url.searchParams.get("publish"));
  const trustRemoteCode = ["1", "true", "yes"].includes(
    String(url.searchParams.get("trust_remote_code") || "").toLowerCase(),
  );
  return { technology, model, targetLayer, device, publish, trustRemoteCode, origin: url.origin };
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
    if (!["GET", "HEAD"].includes(request.method)) {
      return new Response("Método não permitido", {
        status: 405,
        headers: { Allow: "GET, HEAD", "Content-Type": "text/plain; charset=utf-8" },
      });
    }
    try {
      const parameters = requestParameters(request);
      const body = buildLauncher(parameters);
      const filename = `${parameters.technology}-${parameters.model.modelId.replace("/", "-")}.py`;
      return new Response(request.method === "HEAD" ? null : body, {
        status: 200,
        headers: {
          "Cache-Control": "public, max-age=0, s-maxage=300",
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
  normalizeDevice,
  normalizeModel,
  normalizePublish,
  normalizeTargetLayer,
  normalizeTechnology,
  requestParameters,
};
