const REPOSITORY = "programador-powershell/RIFT-LM";
const DEFAULT_REF = "main";
const RESULTS_ENDPOINT = "https://rift-lm.vercel.app/api/results";
const BENCHMARK_PROTOCOL = "LINEAR_REFERENCE_V2";
const TOKENIZER_DEPENDENCIES = {
  transformers: "transformers>=4.52.0",
  accelerate: "accelerate>=0.33.0",
  tokenizers: "tokenizers>=0.20.0",
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
  const device = String(value || "auto").trim().toLowerCase();
  if (!["auto", "cpu", "cuda", "gpu"].includes(device)) {
    throw new ApiError("device precisa ser auto, cpu ou cuda");
  }
  return device === "gpu" ? "auto" : device;
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
  device = "auto",
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
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from urllib.request import Request, urlopen

SCRIPT_URL = ${JSON.stringify(scriptUrl)}
RESULTS_ENDPOINT = ${JSON.stringify(RESULTS_ENDPOINT)}
MODEL_ID = ${JSON.stringify(model.modelId)}
TECHNOLOGY = ${JSON.stringify(definition.label)}
TARGET_LAYER_REQUEST = ${JSON.stringify(targetLayer)}
DEVICE_REQUEST = ${JSON.stringify(device)}
SOURCE_REF = ${JSON.stringify(ref)}
BENCHMARK_PROTOCOL = ${JSON.stringify(BENCHMARK_PROTOCOL)}
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

def gpu_descriptor():
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        rows = [line.strip() for line in output.splitlines() if line.strip()]
        return rows[0] if rows else None
    except (FileNotFoundError, ValueError, subprocess.SubprocessError):
        return None

def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None

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

def report_dependency_state():
    available = []
    missing = []
    for module, package in TOKENIZER_DEPENDENCIES.items():
        try:
            __import__(module)
            available.append(module)
        except Exception:
            missing.append(package)
    if available:
        print("[LAUNCHER] Dependências já presentes:", ", ".join(available))
    if missing:
        print("[LAUNCHER] Dependências ausentes serão instaladas pela própria bateria:", ", ".join(missing))
    # Importante: o launcher NÃO executa pip -U. Cada bateria continua sendo a
    # autoridade sobre suas dependências e evita-se uma instalação duplicada.

def build_comparison_context():
    context = {
        "protocol": BENCHMARK_PROTOCOL,
        "model_id": MODEL_ID,
        "target_layer_request": TARGET_LAYER_REQUEST,
        "device_request": DEVICE_REQUEST,
        "gpu": gpu_descriptor(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "torch": package_version("torch"),
        "transformers": package_version("transformers"),
        "source_ref": SOURCE_REF,
        "context_resolution": "REQUEST_LEVEL",
    }
    # source_ref fica disponível para auditoria, mas não entra no fingerprint:
    # otimizações de uma tecnologia podem ser retestadas contra o mesmo protocolo
    # sem tornar o grupo incompatível só porque houve novo commit do código.
    fingerprint_context = {key: value for key, value in context.items() if key != "source_ref"}
    canonical = json.dumps(fingerprint_context, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    group_id = "cmp-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
    return context, group_id

def install_result_enricher(context, comparison_group_id):
    original_request = urllib.request.Request

    def enriched_request(url, data=None, headers={}, origin_req_host=None, unverifiable=False, method=None):
        target = str(getattr(url, "full_url", url))
        if data is not None and target.rstrip("/") == RESULTS_ENDPOINT.rstrip("/"):
            try:
                payload = json.loads(bytes(data).decode("utf-8"))
                records = payload.get("records") if isinstance(payload, dict) else payload
                if isinstance(records, list):
                    for record in records:
                        if not isinstance(record, dict):
                            continue
                        record.setdefault("schema_version", 1)
                        record.setdefault("technology", TECHNOLOGY)
                        record.setdefault("benchmark_protocol", BENCHMARK_PROTOCOL)
                        record.setdefault("comparison_group_id", comparison_group_id)
                        record.setdefault("comparison_context", context)
                        record.setdefault("implementation", {
                            "scope": "single_linear_reference",
                            "kind": "REFERENCE",
                            "native": False,
                            "simulated": False,
                        })
                    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            except Exception as exc:
                print("[LAUNCHER] AVISO: não foi possível enriquecer o payload de benchmark:", exc)
        return original_request(
            url,
            data=data,
            headers=headers,
            origin_req_host=origin_req_host,
            unverifiable=unverifiable,
            method=method,
        )

    urllib.request.Request = enriched_request

os.environ.setdefault("RIFT_RESULTS_ENDPOINT", RESULTS_ENDPOINT)
os.environ.setdefault("RIFT_SOURCE_REF", SOURCE_REF)
print("[LAUNCHER] Tecnologia: ${definition.label} | Modelo:", MODEL_ID)
for warning in WARNINGS:
    print("[LAUNCHER] AVISO:", warning)
enforce_compatibility()
enforce_publish_settings()
report_dependency_state()
comparison_context, comparison_group_id = build_comparison_context()
os.environ["RIFT_BENCHMARK_PROTOCOL"] = BENCHMARK_PROTOCOL
os.environ["RIFT_COMPARISON_GROUP_ID"] = comparison_group_id
os.environ["RIFT_COMPARISON_CONTEXT_JSON"] = json.dumps(comparison_context, separators=(",", ":"))
install_result_enricher(comparison_context, comparison_group_id)
print("[LAUNCHER] Grupo comparável:", comparison_group_id)
print("[LAUNCHER] Baixando bateria versionada:", SCRIPT_URL)
request = Request(SCRIPT_URL, headers={"User-Agent": "rift-test-launcher/1.1"})
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
  BENCHMARK_PROTOCOL,
  buildLauncher,
  normalizeDevice,
  normalizeModel,
  normalizePublish,
  normalizeTargetLayer,
  normalizeTechnology,
  requestParameters,
};
