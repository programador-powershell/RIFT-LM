// Repo/ref resolvidos server-side (contrato §14.1) — nenhum hardcode de
// owner/repo neste arquivo; o fallback legado vive só em api/_lib/repo.mjs.
import { rawBaseUrl, resolveRef, resolveRepo } from "./_lib/repo.mjs";

const RESULTS_ENDPOINT = "https://rift-lm.vercel.app/api/results";
const BENCHMARK_PROTOCOL = "LINEAR_REFERENCE_V2";
const C3_BENCHMARK_PROTOCOL = "C3_METHODOLOGY_V1";
// Árvore canônica (contrato §20): os scripts multi-motor vivem em batteries/
// no repositório, mas o layout LOCAL no Colab não muda — cada download usa o
// par (caminho_no_repo → caminho_local) e o script cai no diretório de execução.
const C3_SCRIPT = "c3_methodology_auto_batteries.py";
const C3_SCRIPT_REPO_PATH = "batteries/c3_methodology_auto_batteries.py";
const CAP_BENCHMARK_PROTOCOL = "CAPABILITY_PROBE_V1";
const CAP_SCRIPT = "capability_eval_auto_batteries.py";
const CAP_SCRIPT_REPO_PATH = "batteries/capability_eval_auto_batteries.py";
// Caminho GGUF / Muse Glimmer 2-bit no Colab T4 (contrato §11 — GGUF_RUNTIME_V1).
const GGUF_BENCHMARK_PROTOCOL = "GGUF_RUNTIME_V1";
const GGUF_SCRIPT = "gguf_e2e_auto_batteries.py";
const GGUF_SCRIPT_REPO_PATH = "batteries/gguf_e2e_auto_batteries.py";
const GGUF_DEFAULT_QUANT = "UD-Q2_K_XL";
const GGUF_QUANT_RE = /^[A-Za-z0-9_.-]{2,32}$/;
const GGUF_OUTPUT_DIR = "/content/gguf_test_output";
const GGUF_CAP_SERVER_URL = "http://127.0.0.1:8081";
// MicroLM (contrato §22 — 7ª tecnologia, tipo MODELO): a bateria avalia o
// PRÓPRIO modelo de referência (microlm/MicroLM-22M-v0.2) — a rota
// battery=microlm NÃO recebe parâmetro de modelo. O launcher baixa a bateria
// auto-contida E o model.py verbatim de engines/microlm/, pinados no repo/ref
// resolvidos (§14.1).
const MICROLM_BENCHMARK_PROTOCOL = "MICROLM_M0_V1";
const MICROLM_SCRIPT = "microlm_m0_auto_batteries.py";
const MICROLM_SCRIPT_REPO_PATH = "engines/microlm/microlm_m0_auto_batteries.py";
// Pares (caminho_no_repo → caminho_local) como no §20: a bateria importa
// model.py do MESMO diretório de execução.
const MICROLM_MODEL_FILES = [
  ["engines/microlm/model.py", "model.py"],
];
const MICROLM_MODEL_ID = "microlm/MicroLM-22M-v0.2";
// Conversor CASCADE (contrato §26.2 — battery=converter): a célula Colab
// importa os Secrets, instala deps pinadas, baixa o runner auto-contido de
// {origin}/converter.py e o executa com --model [--hf-repo] [--publish on].
// A saída /content/<nome>-cascade é o PRODUTO — NÃO entra na limpeza prévia.
const CONVERTER_BENCHMARK_PROTOCOL = "CONVERTER_STATIC_V1";
const CONVERTER_RUNNER_ROUTE = "/converter.py";
const CONVERTER_RUNNER_LOCAL_PATH = "/content/cascade_converter_runner.py";
const CONVERTER_PIP_PACKAGES = [
  "torch",
  "safetensors>=0.4",
  "numpy>=1.26",
  "huggingface_hub>=0.24,<1",
];
// Mesmo shape org/nome de api/_lib/repo.mjs (REPO_RE) e do runner (MODEL_RE).
const HF_REPO_RE = /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/;
const MICROLM_MODEL = Object.freeze({
  modelId: MICROLM_MODEL_ID,
  trustRemoteCode: false,
  warning: null,
  compatibility: Object.freeze({ colabSupported: true }),
  alias: null,
});
// Limpeza prévia das células Colab geradas (contratos §7, §9, §11, §16 e §22):
// os diretórios de saída do GEYSER, da bateria de capacidades, da bateria
// GGUF, da fase final e do MicroLM constam na lista.
const COLAB_PRECLEAN_PATHS = [
  "/content/geyser_m0_test_output",
  "/content/cap_test_output",
  "/content/gguf_test_output",
  "/content/final_test_output",
  "/content/microlm_m0_test_output",
];
const C3_TECHNOLOGIES = new Set(["rift", "aether", "cascade", "spectra"]);
// Ordem canônica da fila serial C3 quando technology=all (contrato §13.3).
const C3_ALL_TECHNOLOGIES = ["rift", "aether", "cascade", "spectra"];
const WINNER_ARCHITECTURES = new Set(["RIFT", "AETHER", "CASCADE", "SPECTRA"]);
// Lista canônica de arquivos do pacote cascade/ baixados pelas células Colab
// (superconjunto de RIFT_CSERIES_PKG_FILES de api/runner.mjs: inclui também os
// leitores C++ de cascade/runtime/cpp/) + cascade/runtime/cleanup.py.
// Árvore canônica (contrato §20, regra 2): pares [caminho_no_repo, caminho_local].
// O pacote python mora em core/cascade/ no repositório, mas o Colab continua
// baixando para cascade/... locais — os imports das baterias não mudam. O
// conversor tem cópia ÚNICA em core/cascade/converter/ (a duplicata
// cascade-model-converter/ foi eliminada, §20 regra 1).
const C3_PACKAGE_FILES = [
  ["core/cascade/__init__.py", "cascade/__init__.py"],
  ["core/cascade/compiler/__init__.py", "cascade/compiler/__init__.py"],
  ["core/cascade/compiler/cascade_ir.py", "cascade/compiler/cascade_ir.py"],
  ["core/cascade/compiler/decompose.py", "cascade/compiler/decompose.py"],
  ["core/cascade/compiler/bundle_writer.py", "cascade/compiler/bundle_writer.py"],
  ["core/cascade/compiler/block_decompose.py", "cascade/compiler/block_decompose.py"],
  ["core/cascade/kernels/__init__.py", "cascade/kernels/__init__.py"],
  ["core/cascade/kernels/int4.py", "cascade/kernels/int4.py"],
  ["core/cascade/kernels/lowrank.py", "cascade/kernels/lowrank.py"],
  ["core/cascade/kernels/fused_stage.py", "cascade/kernels/fused_stage.py"],
  ["core/cascade/runtime/__init__.py", "cascade/runtime/__init__.py"],
  ["core/cascade/runtime/confidence_gate.py", "cascade/runtime/confidence_gate.py"],
  ["core/cascade/runtime/reference.py", "cascade/runtime/reference.py"],
  ["core/cascade/runtime/block_runtime.py", "cascade/runtime/block_runtime.py"],
  ["core/cascade/runtime/cleanup.py", "cascade/runtime/cleanup.py"],
  // Fontes do leitor C++ mmap (passo 6 — C3_<TECH>_CPP_BUNDLE_READER): sem eles
  // o run_cpp_bundle_reader ficaria SKIPPED para sempre no Colab POSIX com g++.
  ["core/cascade/runtime/cpp/mmap_bundle.hpp", "cascade/runtime/cpp/mmap_bundle.hpp"],
  ["core/cascade/runtime/cpp/mmap_smoke.cpp", "cascade/runtime/cpp/mmap_smoke.cpp"],
  ["core/cascade/converter/__init__.py", "cascade/converter/__init__.py"],
  ["core/cascade/converter/cascade_converter.py", "cascade/converter/cascade_converter.py"],
  ["core/cascade/converter/convert_model.sh", "cascade/converter/convert_model.sh"],
  ["core/cascade/converter/requirements.txt", "cascade/converter/requirements.txt"],
];
// Fase final C4/C5/C6 (contrato §16 — FINAL_PHASE_V1): mesmo estilo da série
// C3 (script auto-contido + pacote cascade/, incl. fontes C++ do leitor mmap),
// mesmas 4 tecnologias; technology=all percorre a fila serial completa.
const FINAL_BENCHMARK_PROTOCOL = "FINAL_PHASE_V1";
const FINAL_SCRIPT = "final_phase_auto_batteries.py";
const FINAL_SCRIPT_REPO_PATH = "batteries/final_phase_auto_batteries.py";
const FINAL_TECHNOLOGIES = C3_TECHNOLOGIES;
const FINAL_ALL_TECHNOLOGIES = C3_ALL_TECHNOLOGIES;
const TOKENIZER_DEPENDENCIES = {
  transformers: "transformers>=4.52.0",
  accelerate: "accelerate>=0.33.0",
  tokenizers: "tokenizers>=0.20.0",
  sentencepiece: "sentencepiece>=0.2.0",
  tiktoken: "tiktoken>=0.7.0",
};

// Árvore canônica (contrato §20): cada bateria M0 vive em engines/<tech>/ no
// repositório; o launcher M0 baixa e executa direto da URL (nada muda no Colab).
const TECHNOLOGIES = {
  rift: {
    label: "RIFT",
    script: "engines/rift/rift_m0_phase1_test_v035_auto_batteries.py",
    arguments: ["--mode", "phase1"],
  },
  cascade: {
    label: "CASCADE",
    script: "engines/cascade/cascade_m0_phase1_test_v030_auto_batteries.py",
    arguments: [],
  },
  aether: {
    label: "AETHER",
    script: "engines/aether/aether_m0_phase1_test_v100_auto_batteries.py",
    arguments: ["--mode", "phase1"],
  },
  spectra: {
    label: "SPECTRA",
    script: "engines/spectra/SPECTRA_Colab_Test_M0.py",
    arguments: ["--mode", "phase1"],
  },
  winner: {
    label: "WINNER",
    script: "engines/winner/winner_m0_phase1_test_v080_auto_batteries.py",
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

function normalizeBattery(value) {
  const battery = String(value || "").trim().toLowerCase();
  if (!battery) return null;
  if (battery === "c3") return "c3";
  if (battery === "cap") return "cap";
  if (battery === "gguf") return "gguf";
  if (battery === "final") return "final";
  if (battery === "microlm") return "microlm";
  if (battery === "converter") return "converter";
  throw new ApiError(
    "battery inválido; use c3, cap ou omita o parâmetro para a bateria M0 "
    + "(gguf dispara a bateria GGUF_RUNTIME_V1 do contrato §11; "
    + "final dispara a fase final FINAL_PHASE_V1 do contrato §16; "
    + "microlm dispara a bateria MICROLM_M0_V1 do contrato §22; "
    + "converter dispara o conversor CASCADE do contrato §26)",
  );
}

/**
 * Conversor (§26.2): hf_repo é OPCIONAL; quando presente precisa ser org/nome
 * ([A-Za-z0-9_.-]) — inválido responde 400 com mensagem clara.
 */
function normalizeHfRepo(value) {
  const hfRepo = String(value || "").trim();
  if (!hfRepo) return null;
  if (!HF_REPO_RE.test(hfRepo)) {
    throw new ApiError(
      "hf_repo inválido; use org/nome (ex.: seu-usuario/meu-modelo-cascade)",
    );
  }
  return hfRepo;
}

/**
 * Conversor (§26.2): publish é on|off (default off) — diferente do
 * normalizePublish das baterias (auto|required|off), porque aqui o publish é
 * repassado ao conversor (--publish) e o padrão do card é NÃO publicar.
 */
function normalizeConverterPublish(value) {
  const mode = String(value || "off").trim().toLowerCase();
  if (!["on", "off"].includes(mode)) {
    throw new ApiError("publish precisa ser on ou off na rota do conversor");
  }
  return mode;
}

// keep_source=on repassa --keep-source-passthrough (§29.3): os tensores fora do
// CASCADE não são copiados e o bundle passa a depender do checkpoint de origem.
function normalizeConverterKeepSource(value) {
  const mode = String(value || "off").trim().toLowerCase();
  if (!["on", "off"].includes(mode)) {
    throw new ApiError("keep_source precisa ser on ou off na rota do conversor");
  }
  return mode;
}

function normalizeQuant(value) {
  const quant = String(value || "").trim();
  if (!quant) return GGUF_DEFAULT_QUANT;
  if (!GGUF_QUANT_RE.test(quant)) {
    throw new ApiError(
      "quant inválido; use de 2 a 32 caracteres [A-Za-z0-9_.-] (ex.: UD-Q2_K_XL)",
    );
  }
  return quant;
}

function normalizeWinnerArch(value) {
  const arch = String(value || "").trim().toUpperCase();
  if (!arch) return null;
  if (!WINNER_ARCHITECTURES.has(arch)) {
    throw new ApiError("arch inválido; use rift, aether, cascade ou spectra");
  }
  return arch;
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
        // Marcador aditivo: a rota battery=gguf (GGUF_RUNTIME_V1, contrato §11)
        // NÃO aplica esta trava — o runtime dela é o llama.cpp, não o Transformers.
        blockKind: "gguf_repo",
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
        // Marcador aditivo: exigência exclusiva dos launchers Transformers;
        // a rota battery=gguf (contrato §11) não aplica esta trava.
        blockKind: "nvfp4_checkpoint",
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

function buildLauncher({
  technology,
  model,
  origin,
  targetLayer = "auto",
  device = "auto",
  publish = "required",
  trustRemoteCode = false,
  winnerArch = null,
  repo = resolveRepo(),
  ref = resolveRef(),
}) {
  const definition = TECHNOLOGIES[technology];
  const winnerArchLine = technology === "winner" && winnerArch
    ? `os.environ["RIFT_WINNER_ARCH"] = ${JSON.stringify(winnerArch)}`
    : "";
  const scriptUrl = `${rawBaseUrl(repo, ref)}/${definition.script}`;
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
GITHUB_REPOSITORY = ${JSON.stringify(repo)}
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
    # O launcher não executa pip -U. A bateria continua sendo a autoridade
    # sobre suas dependências, evitando uma instalação duplicada por execução.

def build_comparison_context(record=None):
    record = record if isinstance(record, dict) else {}
    resolved_layer = record.get("target_layer") or record.get("tensor")
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
        "context_resolution": "RESOLVED" if resolved_layer else "REQUEST_LEVEL",
    }
    if resolved_layer:
        context["target_layer_resolved"] = str(resolved_layer)
    # source_ref é auditável, mas não entra no fingerprint: uma otimização de
    # uma tecnologia pode ser retestada contra o mesmo protocolo sem invalidar
    # o grupo apenas porque houve novo commit do código.
    fingerprint_context = {key: value for key, value in context.items() if key != "source_ref"}
    canonical = json.dumps(fingerprint_context, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    group_id = "cmp-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
    return context, group_id

def implementation_descriptor(record):
    battery = str(record.get("battery_id") or "").upper()
    status = str(record.get("status") or "").upper()
    if status == "SIMULATED" or "_SIM" in battery or "PREFETCH_SIM" in battery or "PIO_POLICY_SIM" in battery:
        return {"scope": "policy_or_reference_simulation", "kind": "SIMULATED", "native": False, "simulated": True}
    if TECHNOLOGY == "WINNER" and ("CPP" in battery or "NATIVE" in battery or "SELF_TEST" in battery):
        return {"scope": "native_cpp_selftest", "kind": "NATIVE", "native": True, "simulated": False}
    return {"scope": "single_linear_reference", "kind": "REFERENCE", "native": False, "simulated": False}

def install_result_enricher():
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
                        existing_context = record.get("comparison_context")
                        has_resolved_context = (
                            isinstance(existing_context, dict)
                            and existing_context.get("context_resolution") == "RESOLVED"
                            and record.get("comparison_group_id")
                        )
                        if not has_resolved_context:
                            context, comparison_group_id = build_comparison_context(record)
                            record["comparison_group_id"] = comparison_group_id
                            record["comparison_context"] = context
                        else:
                            comparison_group_id = str(record["comparison_group_id"])
                            context = existing_context
                        os.environ["RIFT_COMPARISON_GROUP_ID"] = comparison_group_id
                        os.environ["RIFT_COMPARISON_CONTEXT_JSON"] = json.dumps(context, separators=(",", ":"))
                        record.setdefault("schema_version", 1)
                        record.setdefault("technology", TECHNOLOGY)
                        record.setdefault("benchmark_protocol", BENCHMARK_PROTOCOL)
                        record.setdefault("implementation", implementation_descriptor(record))
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
# Repo/ref resolvidos no servidor (contrato §14.2): os scripts Python nunca
# adivinham o repositório — leem estas envs exportadas pela célula gerada.
os.environ.setdefault("RIFT_GITHUB_REPOSITORY", GITHUB_REPOSITORY)
os.environ.setdefault("RIFT_SOURCE_REF", SOURCE_REF)
os.environ["RIFT_BENCHMARK_PROTOCOL"] = BENCHMARK_PROTOCOL
${winnerArchLine}
print("[LAUNCHER] Tecnologia: ${definition.label} | Modelo:", MODEL_ID)
for warning in WARNINGS:
    print("[LAUNCHER] AVISO:", warning)
enforce_compatibility()
enforce_publish_settings()
report_dependency_state()
install_result_enricher()
print("[LAUNCHER] Fingerprint de comparação será finalizado no momento da publicação.")
print("[LAUNCHER] Baixando bateria versionada:", SCRIPT_URL)
request = Request(SCRIPT_URL, headers={"User-Agent": "rift-test-launcher/1.2"})
source = urlopen(request, timeout=60).read()
sys.argv = [SCRIPT_URL, *ARGS]
exec(compile(source, SCRIPT_URL, "exec"), {"__name__": "__main__", "__file__": SCRIPT_URL})
`;
}

/**
 * Fila serial C3 (contrato §13.3): technology=all gera UMA célula que roda a
 * metodologia de 16 passos para rift, aether, cascade e spectra em sequência.
 * A UI só expõe "all"; as rotas por tecnologia seguem aceitas para automação.
 */
function buildC3AllLauncher({
  model,
  origin,
  publish = "required",
  repo = resolveRepo(),
  ref = resolveRef(),
}) {
  const repoBase = rawBaseUrl(repo, ref);
  const compatibility = model.compatibility || { colabSupported: true };
  return `#!/usr/bin/env python3
# Launcher C3_METHODOLOGY_V1 gerado por ${origin} para TODAS as tecnologias
# (fila serial completa: rift -> aether -> cascade -> spectra; 16 passos cada).
# O benchmark usa GPU do Colab; a Vercel apenas entrega este inicializador.
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.request import Request, urlopen

REPO_BASE = ${JSON.stringify(repoBase)}
RESULTS_ENDPOINT = ${JSON.stringify(RESULTS_ENDPOINT)}
BENCHMARK_PROTOCOL = ${JSON.stringify(C3_BENCHMARK_PROTOCOL)}
GITHUB_REPOSITORY = ${JSON.stringify(repo)}
SOURCE_REF = ${JSON.stringify(ref)}
TECHNOLOGIES = json.loads(r'''${JSON.stringify(C3_ALL_TECHNOLOGIES)}''')
MODEL_ID = ${JSON.stringify(model.modelId)}
PUBLISH_MODE = ${JSON.stringify(publish)}
SCRIPT_NAME = ${JSON.stringify(C3_SCRIPT)}
SCRIPT_REPO_PATH = ${JSON.stringify(C3_SCRIPT_REPO_PATH)}
PACKAGE_FILES = json.loads(r'''${JSON.stringify(C3_PACKAGE_FILES)}''')
COMPATIBILITY = json.loads(r'''${JSON.stringify(compatibility)}''')

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

def enforce_compatibility():
    if COMPATIBILITY.get("colabSupported", True):
        return
    print("[C3] BLOQUEADO: este modelo não é compatível com a bateria Colab atual.")
    print("[C3] Motivo:", COMPATIBILITY.get("reason", "Recursos insuficientes."))
    print("[C3] Nenhum peso foi baixado e nenhum resultado foi publicado.")
    raise SystemExit(2)

def enforce_transport_security():
    endpoint = os.environ.get("RIFT_RESULTS_ENDPOINT", RESULTS_ENDPOINT).strip()
    if not endpoint.lower().startswith("https://"):
        print("[C3] ERRO: RIFT_RESULTS_ENDPOINT precisa usar HTTPS.")
        raise SystemExit(2)
    if not REPO_BASE.lower().startswith("https://"):
        print("[C3] ERRO: o download da bateria precisa usar HTTPS.")
        raise SystemExit(2)

def enforce_publish_settings():
    if PUBLISH_MODE == "off":
        return
    token = os.environ.get("RIFT_INGEST_TOKEN", "").strip()
    if not token:
        print("[C3] ERRO: RIFT_INGEST_TOKEN não chegou ao kernel do Colab.")
        print("[C3] Cadastre o Secret RIFT_INGEST_TOKEN no Colab e rode a célula novamente.")
        raise SystemExit(2)
    if len(token) < 32:
        print("[C3] ERRO: RIFT_INGEST_TOKEN precisa ter pelo menos 32 caracteres.")
        raise SystemExit(2)

ROOT = Path("/content/c3_run") if os.path.isdir("/content") else Path.cwd() / "c3_run"

def download(repo_path, local_path):
    # Par (caminho_no_repo -> caminho_local) do contrato §20: o layout local
    # do Colab não muda mesmo com a árvore canônica do repositório.
    url = REPO_BASE + "/" + repo_path
    destination = ROOT / local_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "rift-c3-launcher/1.0"})
    destination.write_bytes(urlopen(request, timeout=60).read())
    print("[C3] download", repo_path, "->", local_path)

import_colab_secrets()
enforce_compatibility()
enforce_transport_security()
enforce_publish_settings()
os.environ.setdefault("RIFT_RESULTS_ENDPOINT", RESULTS_ENDPOINT)
os.environ.setdefault("RIFT_GITHUB_REPOSITORY", GITHUB_REPOSITORY)
os.environ.setdefault("RIFT_SOURCE_REF", SOURCE_REF)
os.environ["RIFT_BENCHMARK_PROTOCOL"] = BENCHMARK_PROTOCOL
ROOT.mkdir(parents=True, exist_ok=True)
print("[C3] Fila serial completa:", ", ".join(TECHNOLOGIES), "| Modelo:", MODEL_ID, "| ref:", SOURCE_REF)
download(SCRIPT_REPO_PATH, SCRIPT_NAME)
for repo_path, local_path in PACKAGE_FILES:
    download(repo_path, local_path)
results = {}
for technology in TECHNOLOGIES:
    print("[C3] executando", SCRIPT_NAME, "--technology", technology, "--model", MODEL_ID)
    command = [sys.executable, str(ROOT / SCRIPT_NAME), "--technology", technology, "--model", MODEL_ID]
    results[technology] = subprocess.call(command, cwd=str(ROOT), env=os.environ.copy())
    print("[C3]", technology, "finalizado rc =", results[technology])
failures = {technology: rc for technology, rc in results.items() if rc != 0}
print("[C3] fila serial finalizada:", json.dumps(results, sort_keys=True))
if failures:
    print("[C3] FALHAS:", json.dumps(failures, sort_keys=True))
    raise SystemExit(1)
`;
}

function buildC3Launcher({
  technology,
  model,
  origin,
  publish = "required",
  repo = resolveRepo(),
  ref = resolveRef(),
}) {
  if (technology === "all") {
    return buildC3AllLauncher({ model, origin, publish, repo, ref });
  }
  if (!C3_TECHNOLOGIES.has(technology)) {
    throw new ApiError(
      "A série C3 aceita apenas rift, aether, cascade ou spectra "
      + "(ou all para a fila serial completa)",
    );
  }
  const definition = TECHNOLOGIES[technology];
  const repoBase = rawBaseUrl(repo, ref);
  const compatibility = model.compatibility || { colabSupported: true };
  return `#!/usr/bin/env python3
# Launcher C3_METHODOLOGY_V1 gerado por ${origin} para ${definition.label} (16 passos).
# O benchmark usa GPU do Colab; a Vercel apenas entrega este inicializador.
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.request import Request, urlopen

REPO_BASE = ${JSON.stringify(repoBase)}
RESULTS_ENDPOINT = ${JSON.stringify(RESULTS_ENDPOINT)}
BENCHMARK_PROTOCOL = ${JSON.stringify(C3_BENCHMARK_PROTOCOL)}
GITHUB_REPOSITORY = ${JSON.stringify(repo)}
SOURCE_REF = ${JSON.stringify(ref)}
TECHNOLOGY = ${JSON.stringify(technology)}
MODEL_ID = ${JSON.stringify(model.modelId)}
PUBLISH_MODE = ${JSON.stringify(publish)}
SCRIPT_NAME = ${JSON.stringify(C3_SCRIPT)}
SCRIPT_REPO_PATH = ${JSON.stringify(C3_SCRIPT_REPO_PATH)}
PACKAGE_FILES = json.loads(r'''${JSON.stringify(C3_PACKAGE_FILES)}''')
COMPATIBILITY = json.loads(r'''${JSON.stringify(compatibility)}''')
ARGS = ["--technology", TECHNOLOGY, "--model", MODEL_ID]

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

def enforce_compatibility():
    if COMPATIBILITY.get("colabSupported", True):
        return
    print("[C3] BLOQUEADO: este modelo não é compatível com a bateria Colab atual.")
    print("[C3] Motivo:", COMPATIBILITY.get("reason", "Recursos insuficientes."))
    print("[C3] Nenhum peso foi baixado e nenhum resultado foi publicado.")
    raise SystemExit(2)

def enforce_transport_security():
    endpoint = os.environ.get("RIFT_RESULTS_ENDPOINT", RESULTS_ENDPOINT).strip()
    if not endpoint.lower().startswith("https://"):
        print("[C3] ERRO: RIFT_RESULTS_ENDPOINT precisa usar HTTPS.")
        raise SystemExit(2)
    if not REPO_BASE.lower().startswith("https://"):
        print("[C3] ERRO: o download da bateria precisa usar HTTPS.")
        raise SystemExit(2)

def enforce_publish_settings():
    if PUBLISH_MODE == "off":
        return
    token = os.environ.get("RIFT_INGEST_TOKEN", "").strip()
    if not token:
        print("[C3] ERRO: RIFT_INGEST_TOKEN não chegou ao kernel do Colab.")
        print("[C3] Cadastre o Secret RIFT_INGEST_TOKEN no Colab e rode a célula novamente.")
        raise SystemExit(2)
    if len(token) < 32:
        print("[C3] ERRO: RIFT_INGEST_TOKEN precisa ter pelo menos 32 caracteres.")
        raise SystemExit(2)

ROOT = Path("/content/c3_run") if os.path.isdir("/content") else Path.cwd() / "c3_run"

def download(repo_path, local_path):
    # Par (caminho_no_repo -> caminho_local) do contrato §20: o layout local
    # do Colab não muda mesmo com a árvore canônica do repositório.
    url = REPO_BASE + "/" + repo_path
    destination = ROOT / local_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "rift-c3-launcher/1.0"})
    destination.write_bytes(urlopen(request, timeout=60).read())
    print("[C3] download", repo_path, "->", local_path)

import_colab_secrets()
enforce_compatibility()
enforce_transport_security()
enforce_publish_settings()
os.environ.setdefault("RIFT_RESULTS_ENDPOINT", RESULTS_ENDPOINT)
os.environ.setdefault("RIFT_GITHUB_REPOSITORY", GITHUB_REPOSITORY)
os.environ.setdefault("RIFT_SOURCE_REF", SOURCE_REF)
os.environ["RIFT_BENCHMARK_PROTOCOL"] = BENCHMARK_PROTOCOL
ROOT.mkdir(parents=True, exist_ok=True)
print("[C3] Tecnologia:", TECHNOLOGY, "| Modelo:", MODEL_ID, "| ref:", SOURCE_REF)
download(SCRIPT_REPO_PATH, SCRIPT_NAME)
for repo_path, local_path in PACKAGE_FILES:
    download(repo_path, local_path)
print("[C3] executando", SCRIPT_NAME, "--technology", TECHNOLOGY, "--model", MODEL_ID)
command = [sys.executable, str(ROOT / SCRIPT_NAME), *ARGS]
return_code = subprocess.call(command, cwd=str(ROOT), env=os.environ.copy())
print("[C3] finalizado rc =", return_code)
if return_code != 0:
    raise SystemExit(return_code)
`;
}

/**
 * Fase final FINAL_PHASE_V1 (contrato §16 — C4/C5/C6 até o marco
 * "compilador e executor LLM"): battery=final gera UMA célula Colab que baixa
 * final_phase_auto_batteries.py + o MESMO pacote cascade/ do launcher C3
 * (incl. fontes C++ do leitor mmap), pinados no repo/ref resolvidos (§14.1),
 * e roda --technology <tech> --model <model>. technology=all percorre as 4
 * tecnologias em fila serial (rift -> aether -> cascade -> spectra).
 */
function buildFinalLauncher({
  technology,
  model,
  origin,
  publish = "required",
  repo = resolveRepo(),
  ref = resolveRef(),
}) {
  if (technology !== "all" && !FINAL_TECHNOLOGIES.has(technology)) {
    throw new ApiError(
      "A fase final aceita apenas rift, aether, cascade ou spectra "
      + "(ou all para a fila serial completa)",
    );
  }
  const technologies = technology === "all" ? FINAL_ALL_TECHNOLOGIES : [technology];
  const repoBase = rawBaseUrl(repo, ref);
  const compatibility = model.compatibility || { colabSupported: true };
  return `#!/usr/bin/env python3
# Launcher FINAL_PHASE_V1 gerado por ${origin} (fase final C4/C5/C6, contrato §16)
# — tecnologia(s): ${technologies.join(" -> ")}.
# O benchmark usa GPU do Colab; a Vercel apenas entrega este inicializador.
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import Request, urlopen

REPO_BASE = ${JSON.stringify(repoBase)}
RESULTS_ENDPOINT = ${JSON.stringify(RESULTS_ENDPOINT)}
BENCHMARK_PROTOCOL = ${JSON.stringify(FINAL_BENCHMARK_PROTOCOL)}
GITHUB_REPOSITORY = ${JSON.stringify(repo)}
SOURCE_REF = ${JSON.stringify(ref)}
TECHNOLOGIES = json.loads(r'''${JSON.stringify(technologies)}''')
MODEL_ID = ${JSON.stringify(model.modelId)}
PUBLISH_MODE = ${JSON.stringify(publish)}
SCRIPT_NAME = ${JSON.stringify(FINAL_SCRIPT)}
SCRIPT_REPO_PATH = ${JSON.stringify(FINAL_SCRIPT_REPO_PATH)}
PACKAGE_FILES = json.loads(r'''${JSON.stringify(C3_PACKAGE_FILES)}''')
PRECLEAN_PATHS = json.loads(r'''${JSON.stringify(COLAB_PRECLEAN_PATHS)}''')
COMPATIBILITY = json.loads(r'''${JSON.stringify(compatibility)}''')

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

def enforce_compatibility():
    if COMPATIBILITY.get("colabSupported", True):
        return
    print("[FINAL] BLOQUEADO: este modelo não é compatível com a bateria Colab atual.")
    print("[FINAL] Motivo:", COMPATIBILITY.get("reason", "Recursos insuficientes."))
    print("[FINAL] Nenhum peso foi baixado e nenhum resultado foi publicado.")
    raise SystemExit(2)

def enforce_transport_security():
    endpoint = os.environ.get("RIFT_RESULTS_ENDPOINT", RESULTS_ENDPOINT).strip()
    if not endpoint.lower().startswith("https://"):
        print("[FINAL] ERRO: RIFT_RESULTS_ENDPOINT precisa usar HTTPS.")
        raise SystemExit(2)
    if not REPO_BASE.lower().startswith("https://"):
        print("[FINAL] ERRO: o download da bateria precisa usar HTTPS.")
        raise SystemExit(2)

def enforce_publish_settings():
    if PUBLISH_MODE == "off":
        return
    token = os.environ.get("RIFT_INGEST_TOKEN", "").strip()
    if not token:
        print("[FINAL] ERRO: RIFT_INGEST_TOKEN não chegou ao kernel do Colab.")
        print("[FINAL] Cadastre o Secret RIFT_INGEST_TOKEN no Colab e rode a célula novamente.")
        raise SystemExit(2)
    if len(token) < 32:
        print("[FINAL] ERRO: RIFT_INGEST_TOKEN precisa ter pelo menos 32 caracteres.")
        raise SystemExit(2)

def preclean_workspace():
    # Limpeza destrutiva só no Colab e só sob /content (contrato §5).
    if not os.path.isdir("/content"):
        return
    for path in PRECLEAN_PATHS:
        if not str(path).startswith("/content/"):
            continue
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
            print("[FINAL] limpeza prévia:", path)

ROOT = Path("/content/final_run") if os.path.isdir("/content") else Path.cwd() / "final_run"

def download(repo_path, local_path):
    # Par (caminho_no_repo -> caminho_local) do contrato §20: o layout local
    # do Colab não muda mesmo com a árvore canônica do repositório.
    url = REPO_BASE + "/" + repo_path
    destination = ROOT / local_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "rift-final-launcher/1.0"})
    destination.write_bytes(urlopen(request, timeout=60).read())
    print("[FINAL] download", repo_path, "->", local_path)

import_colab_secrets()
enforce_compatibility()
enforce_transport_security()
enforce_publish_settings()
preclean_workspace()
os.environ.setdefault("RIFT_RESULTS_ENDPOINT", RESULTS_ENDPOINT)
os.environ.setdefault("RIFT_GITHUB_REPOSITORY", GITHUB_REPOSITORY)
os.environ.setdefault("RIFT_SOURCE_REF", SOURCE_REF)
os.environ["RIFT_BENCHMARK_PROTOCOL"] = BENCHMARK_PROTOCOL
ROOT.mkdir(parents=True, exist_ok=True)
print("[FINAL] Fase final C4/C5/C6:", ", ".join(TECHNOLOGIES), "| Modelo:", MODEL_ID, "| ref:", SOURCE_REF)
download(SCRIPT_REPO_PATH, SCRIPT_NAME)
for repo_path, local_path in PACKAGE_FILES:
    download(repo_path, local_path)
results = {}
for technology in TECHNOLOGIES:
    print("[FINAL] executando", SCRIPT_NAME, "--technology", technology, "--model", MODEL_ID)
    command = [sys.executable, str(ROOT / SCRIPT_NAME), "--technology", technology, "--model", MODEL_ID]
    results[technology] = subprocess.call(command, cwd=str(ROOT), env=os.environ.copy())
    print("[FINAL]", technology, "finalizado rc =", results[technology])
failures = {technology: rc for technology, rc in results.items() if rc != 0}
print("[FINAL] fase final finalizada:", json.dumps(results, sort_keys=True))
if failures:
    print("[FINAL] FALHAS:", json.dumps(failures, sort_keys=True))
    raise SystemExit(1)
`;
}

function buildCapLauncher({
  model,
  origin,
  publish = "required",
  repo = resolveRepo(),
  ref = resolveRef(),
}) {
  // Árvore canônica (contrato §20): a bateria de capacidades vive em batteries/.
  const scriptUrl = `${rawBaseUrl(repo, ref)}/${CAP_SCRIPT_REPO_PATH}`;
  const compatibility = model.compatibility || { colabSupported: true };
  return `#!/usr/bin/env python3
# Launcher CAPABILITY_PROBE_V1 gerado por ${origin} (bateria de capacidades, §9).
# Probe leve embutido — não é MMLU/HumanEval/SWE-bench completos.
# O benchmark usa GPU do Colab; a Vercel apenas entrega este inicializador.
import json
import os
import shutil
import sys
from urllib.request import Request, urlopen

SCRIPT_URL = ${JSON.stringify(scriptUrl)}
RESULTS_ENDPOINT = ${JSON.stringify(RESULTS_ENDPOINT)}
BENCHMARK_PROTOCOL = ${JSON.stringify(CAP_BENCHMARK_PROTOCOL)}
GITHUB_REPOSITORY = ${JSON.stringify(repo)}
SOURCE_REF = ${JSON.stringify(ref)}
MODEL_ID = ${JSON.stringify(model.modelId)}
PUBLISH_MODE = ${JSON.stringify(publish)}
PRECLEAN_PATHS = json.loads(r'''${JSON.stringify(COLAB_PRECLEAN_PATHS)}''')
COMPATIBILITY = json.loads(r'''${JSON.stringify(compatibility)}''')
ARGS = ["--model", MODEL_ID]

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

def enforce_compatibility():
    if COMPATIBILITY.get("colabSupported", True):
        return
    print("[CAP] BLOQUEADO: este modelo não é compatível com a bateria Colab atual.")
    print("[CAP] Motivo:", COMPATIBILITY.get("reason", "Recursos insuficientes."))
    print("[CAP] Nenhum peso foi baixado e nenhum resultado foi publicado.")
    raise SystemExit(2)

def enforce_transport_security():
    endpoint = os.environ.get("RIFT_RESULTS_ENDPOINT", RESULTS_ENDPOINT).strip()
    if not endpoint.lower().startswith("https://"):
        print("[CAP] ERRO: RIFT_RESULTS_ENDPOINT precisa usar HTTPS.")
        raise SystemExit(2)
    if not SCRIPT_URL.lower().startswith("https://"):
        print("[CAP] ERRO: o download da bateria precisa usar HTTPS.")
        raise SystemExit(2)

def enforce_publish_settings():
    if PUBLISH_MODE == "off":
        return
    token = os.environ.get("RIFT_INGEST_TOKEN", "").strip()
    if not token:
        print("[CAP] ERRO: RIFT_INGEST_TOKEN não chegou ao kernel do Colab.")
        print("[CAP] Cadastre o Secret RIFT_INGEST_TOKEN no Colab e rode a célula novamente.")
        raise SystemExit(2)
    if len(token) < 32:
        print("[CAP] ERRO: RIFT_INGEST_TOKEN precisa ter pelo menos 32 caracteres.")
        raise SystemExit(2)

def preclean_workspace():
    # Limpeza destrutiva só no Colab e só sob /content (contrato §5).
    if not os.path.isdir("/content"):
        return
    for path in PRECLEAN_PATHS:
        if not str(path).startswith("/content/"):
            continue
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
            print("[CAP] limpeza prévia:", path)

import_colab_secrets()
enforce_compatibility()
enforce_transport_security()
enforce_publish_settings()
preclean_workspace()
os.environ.setdefault("RIFT_RESULTS_ENDPOINT", RESULTS_ENDPOINT)
os.environ.setdefault("RIFT_GITHUB_REPOSITORY", GITHUB_REPOSITORY)
os.environ.setdefault("RIFT_SOURCE_REF", SOURCE_REF)
os.environ["RIFT_BENCHMARK_PROTOCOL"] = BENCHMARK_PROTOCOL
print("[CAP] Modelo:", MODEL_ID, "| ref:", SOURCE_REF)
print("[CAP] Baixando bateria versionada:", SCRIPT_URL)
request = Request(SCRIPT_URL, headers={"User-Agent": "rift-cap-launcher/1.0"})
source = urlopen(request, timeout=60).read()
sys.argv = [SCRIPT_URL, *ARGS]
exec(compile(source, SCRIPT_URL, "exec"), {"__name__": "__main__", "__file__": SCRIPT_URL})
`;
}

/**
 * Exceção do contrato §11: a trava anti-GGUF/anti-NVFP4 dos launchers baseados
 * em Transformers NÃO se aplica à rota battery=gguf — o runtime dela é o
 * llama.cpp. Os demais bloqueios (ex.: Kimi-K3/DeepSeek-V4 por tamanho total)
 * continuam valendo também neste caminho.
 */
function ggufRuntimeModel(model) {
  const compatibility = model.compatibility || { colabSupported: true };
  if (compatibility.colabSupported !== false) return model;
  if (!["gguf_repo", "nvfp4_checkpoint"].includes(compatibility.blockKind)) return model;
  return {
    ...model,
    compatibility: {
      colabSupported: true,
      note: (
        "GGUF_RUNTIME_V1 (contrato §11): a trava anti-GGUF dos launchers "
        + "Transformers não se aplica a esta rota; o runtime é o llama.cpp."
      ),
    },
  };
}

function buildGgufLauncher({
  model,
  origin,
  quant = GGUF_DEFAULT_QUANT,
  publish = "required",
  repo = resolveRepo(),
  ref = resolveRef(),
}) {
  const repoBase = rawBaseUrl(repo, ref);
  const compatibility = model.compatibility || { colabSupported: true };
  return `#!/usr/bin/env python3
# Launcher GGUF_RUNTIME_V1 gerado por ${origin} (contrato §11 — Muse Glimmer 2-bit / caminho GGUF).
# O benchmark roda no Colab (T4, llama.cpp com offload parcial CPU/GPU); a Vercel
# apenas entrega este inicializador.
# O download do binário llama.cpp (release oficial PINADA por tag + sha256) e dos
# arquivos .gguf do quant escolhido acontece DENTRO de gguf_e2e_auto_batteries.py —
# este launcher não instala dependência alguma; ele só baixa os scripts versionados.
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

REPO_BASE = ${JSON.stringify(repoBase)}
RESULTS_ENDPOINT = ${JSON.stringify(RESULTS_ENDPOINT)}
BENCHMARK_PROTOCOL = ${JSON.stringify(GGUF_BENCHMARK_PROTOCOL)}
GITHUB_REPOSITORY = ${JSON.stringify(repo)}
SOURCE_REF = ${JSON.stringify(ref)}
MODEL_ID = ${JSON.stringify(model.modelId)}
QUANT = ${JSON.stringify(quant)}
PUBLISH_MODE = ${JSON.stringify(publish)}
GGUF_SCRIPT = ${JSON.stringify(GGUF_SCRIPT)}
GGUF_SCRIPT_REPO_PATH = ${JSON.stringify(GGUF_SCRIPT_REPO_PATH)}
CAP_SCRIPT = ${JSON.stringify(CAP_SCRIPT)}
CAP_SCRIPT_REPO_PATH = ${JSON.stringify(CAP_SCRIPT_REPO_PATH)}
OUTPUT_DIR = ${JSON.stringify(GGUF_OUTPUT_DIR)} if os.path.isdir("/content") else str(Path.cwd() / "gguf_run" / "gguf_test_output")
CAP_SERVER_URL = ${JSON.stringify(GGUF_CAP_SERVER_URL)}
PRECLEAN_PATHS = json.loads(r'''${JSON.stringify(COLAB_PRECLEAN_PATHS)}''')
COMPATIBILITY = json.loads(r'''${JSON.stringify(compatibility)}''')
# --out explícito: sem ele a bateria resolveria o default relativo contra o cwd do
# subprocesso (ROOT) e os artefatos (llama.cpp + .gguf, ~12-15 GB) escapariam do
# diretório varrido pelo CAP probe e pela limpeza prévia (contrato §11).
GGUF_ARGS = ["--model", MODEL_ID, "--quant", QUANT, "--out", OUTPUT_DIR]
# --model-id-label: registros CAP-sobre-GGUF usam model_id '<repo>:<quant>'
# (contrato §11) para distinguir quants do mesmo repositório GGUF.
CAP_ARGS = ["--model", MODEL_ID, "--model-id-label", MODEL_ID + ":" + QUANT, "--backend", "llamacpp", "--server-url", CAP_SERVER_URL]
if PUBLISH_MODE == "off":
    # ?publish=off vale para a bateria E para o probe (o default dos scripts é on).
    GGUF_ARGS += ["--publish", "off"]
    CAP_ARGS += ["--publish", "off"]

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

def enforce_compatibility():
    if COMPATIBILITY.get("colabSupported", True):
        return
    print("[GGUF] BLOQUEADO: este modelo não é compatível com a bateria Colab atual.")
    print("[GGUF] Motivo:", COMPATIBILITY.get("reason", "Recursos insuficientes."))
    print("[GGUF] Nenhum peso foi baixado e nenhum resultado foi publicado.")
    raise SystemExit(2)

def enforce_transport_security():
    endpoint = os.environ.get("RIFT_RESULTS_ENDPOINT", RESULTS_ENDPOINT).strip()
    if not endpoint.lower().startswith("https://"):
        print("[GGUF] ERRO: RIFT_RESULTS_ENDPOINT precisa usar HTTPS.")
        raise SystemExit(2)
    if not REPO_BASE.lower().startswith("https://"):
        print("[GGUF] ERRO: o download da bateria precisa usar HTTPS.")
        raise SystemExit(2)

def enforce_publish_settings():
    if PUBLISH_MODE == "off":
        return
    token = os.environ.get("RIFT_INGEST_TOKEN", "").strip()
    if not token:
        print("[GGUF] ERRO: RIFT_INGEST_TOKEN não chegou ao kernel do Colab.")
        print("[GGUF] Cadastre o Secret RIFT_INGEST_TOKEN no Colab e rode a célula novamente.")
        raise SystemExit(2)
    if len(token) < 32:
        print("[GGUF] ERRO: RIFT_INGEST_TOKEN precisa ter pelo menos 32 caracteres.")
        raise SystemExit(2)

def preclean_workspace():
    # Limpeza destrutiva só no Colab e só sob /content (contrato §5).
    if not os.path.isdir("/content"):
        return
    for path in PRECLEAN_PATHS:
        if not str(path).startswith("/content/"):
            continue
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
            print("[GGUF] limpeza prévia:", path)

ROOT = Path("/content/gguf_run") if os.path.isdir("/content") else Path.cwd() / "gguf_run"

def download(repo_path, local_path):
    # Par (caminho_no_repo -> caminho_local) do contrato §20: o layout local
    # do Colab não muda mesmo com a árvore canônica do repositório.
    url = REPO_BASE + "/" + repo_path
    destination = ROOT / local_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "rift-gguf-launcher/1.0"})
    destination.write_bytes(urlopen(request, timeout=60).read())
    print("[GGUF] download", repo_path, "->", local_path)

def find_llamacpp_runtime():
    # Convenção com a bateria: gguf_e2e_auto_batteries.py deixa os artefatos em
    # OUTPUT_DIR e PODE gravar OUTPUT_DIR/llamacpp_runtime.json com
    # {"server_bin": ..., "gguf_path": ..., "n_gpu_layers": N}. Sem o arquivo,
    # o launcher procura o binário llama-server e o .gguf do quant no diretório.
    server_bin = None
    gguf_path = None
    ngl = os.environ.get("RIFT_GGUF_NGL", "").strip() or None
    state_path = Path(OUTPUT_DIR) / "llamacpp_runtime.json"
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            server_bin = state.get("server_bin") or None
            gguf_path = state.get("gguf_path") or None
            if ngl is None and state.get("n_gpu_layers") is not None:
                ngl = str(state.get("n_gpu_layers"))
        except Exception as exc:
            print("[GGUF] AVISO: llamacpp_runtime.json ilegível:", exc)
    if not server_bin:
        server_bin = shutil.which("llama-server")
    if not server_bin and os.path.isdir(OUTPUT_DIR):
        candidates = sorted(Path(OUTPUT_DIR).rglob("llama-server"))
        if candidates:
            server_bin = str(candidates[0])
    if not gguf_path and os.path.isdir(OUTPUT_DIR):
        ggufs = sorted(Path(OUTPUT_DIR).rglob("*.gguf"))
        preferred = [item for item in ggufs if QUANT.lower() in item.name.lower()]
        chosen = preferred or ggufs
        if chosen:
            gguf_path = str(chosen[0])
    return server_bin, gguf_path, ngl

def wait_for_health(url, timeout_s=180):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            request = Request(url, headers={"User-Agent": "rift-gguf-launcher/1.0"})
            with urlopen(request, timeout=5) as response:
                if 200 <= getattr(response, "status", 200) < 300:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False

def run_cap_probe():
    # Passo OPCIONAL (best-effort): CAP_* via llama-server (API OpenAI-compatível,
    # contrato §11). Qualquer falha aqui vira aviso e NÃO altera o código de saída
    # da célula — o resultado da bateria GGUF já está garantido.
    if os.environ.get("RIFT_GGUF_CAP", "1").strip().lower() in ("0", "false", "off"):
        print("[GGUF] CAP probe desativado por RIFT_GGUF_CAP=0.")
        return
    server_bin, gguf_path, ngl = find_llamacpp_runtime()
    if not server_bin or not gguf_path:
        print("[GGUF] CAP probe pulado: llama-server ou .gguf não encontrados em", OUTPUT_DIR)
        return
    command = [server_bin, "-m", gguf_path, "--host", "127.0.0.1", "--port", "8081"]
    if ngl is not None:
        command += ["-ngl", str(ngl)]
    print("[GGUF] iniciando llama-server para o CAP probe:", " ".join(command))
    server = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        if not wait_for_health(CAP_SERVER_URL + "/health"):
            print("[GGUF] CAP probe pulado: llama-server não respondeu em /health.")
            return
        print("[GGUF] executando", CAP_SCRIPT, "--backend llamacpp --server-url", CAP_SERVER_URL)
        cap_rc = subprocess.call(
            [sys.executable, str(ROOT / CAP_SCRIPT), *CAP_ARGS],
            cwd=str(ROOT),
            env=os.environ.copy(),
        )
        print("[GGUF] CAP probe rc =", cap_rc)
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()

import_colab_secrets()
enforce_compatibility()
enforce_transport_security()
enforce_publish_settings()
preclean_workspace()
os.environ.setdefault("RIFT_RESULTS_ENDPOINT", RESULTS_ENDPOINT)
os.environ.setdefault("RIFT_GITHUB_REPOSITORY", GITHUB_REPOSITORY)
os.environ.setdefault("RIFT_SOURCE_REF", SOURCE_REF)
os.environ["RIFT_BENCHMARK_PROTOCOL"] = BENCHMARK_PROTOCOL
ROOT.mkdir(parents=True, exist_ok=True)
print("[GGUF] Modelo:", MODEL_ID, "| quant:", QUANT, "| ref:", SOURCE_REF)
download(GGUF_SCRIPT_REPO_PATH, GGUF_SCRIPT)
download(CAP_SCRIPT_REPO_PATH, CAP_SCRIPT)
print("[GGUF] executando", GGUF_SCRIPT, "--model", MODEL_ID, "--quant", QUANT)
return_code = subprocess.call(
    [sys.executable, str(ROOT / GGUF_SCRIPT), *GGUF_ARGS],
    cwd=str(ROOT),
    env=os.environ.copy(),
)
print("[GGUF] bateria finalizada rc =", return_code)
if return_code != 0:
    raise SystemExit(return_code)
try:
    run_cap_probe()
except Exception as exc:
    print("[GGUF] AVISO: CAP probe falhou sem afetar a bateria GGUF:", exc)
print("[GGUF] finalizado.")
`;
}

/**
 * MicroLM (contrato §22 — MICROLM_M0_V1): battery=microlm gera UMA célula
 * Colab com o bootstrap de segurança padrão (Secrets do Colab, HTTPS
 * obrigatório, token >= 32, limpeza prévia só sob /content) que baixa a
 * bateria auto-contida engines/microlm/microlm_m0_auto_batteries.py E o
 * engines/microlm/model.py verbatim (a bateria importa model.py do mesmo
 * diretório), pinados no repo/ref resolvidos (§14.1), e a executa em
 * subprocesso. Não há parâmetro de modelo: o MicroLM É o modelo (referência
 * fixa microlm/MicroLM-22M-v0.2); torch é obrigatório e CPU é suficiente.
 */
function buildMicrolmLauncher({
  origin,
  publish = "required",
  repo = resolveRepo(),
  ref = resolveRef(),
}) {
  const repoBase = rawBaseUrl(repo, ref);
  return `#!/usr/bin/env python3
# Launcher MICROLM_M0_V1 gerado por ${origin} (contrato §22 — MicroLM, 7ª
# tecnologia, tipo MODELO). A bateria avalia o próprio modelo de referência
# (~22M params ativos); torch é obrigatório e CPU é suficiente. A Vercel
# apenas entrega este inicializador.
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import Request, urlopen

REPO_BASE = ${JSON.stringify(repoBase)}
RESULTS_ENDPOINT = ${JSON.stringify(RESULTS_ENDPOINT)}
BENCHMARK_PROTOCOL = ${JSON.stringify(MICROLM_BENCHMARK_PROTOCOL)}
GITHUB_REPOSITORY = ${JSON.stringify(repo)}
SOURCE_REF = ${JSON.stringify(ref)}
MODEL_ID = ${JSON.stringify(MICROLM_MODEL_ID)}
PUBLISH_MODE = ${JSON.stringify(publish)}
SCRIPT_NAME = ${JSON.stringify(MICROLM_SCRIPT)}
SCRIPT_REPO_PATH = ${JSON.stringify(MICROLM_SCRIPT_REPO_PATH)}
MODEL_FILES = json.loads(r'''${JSON.stringify(MICROLM_MODEL_FILES)}''')
PRECLEAN_PATHS = json.loads(r'''${JSON.stringify(COLAB_PRECLEAN_PATHS)}''')
ARGS = []
if PUBLISH_MODE == "off":
    # ?publish=off é repassado à bateria (o default do script é publicar).
    ARGS += ["--publish", "off"]

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

def enforce_transport_security():
    endpoint = os.environ.get("RIFT_RESULTS_ENDPOINT", RESULTS_ENDPOINT).strip()
    if not endpoint.lower().startswith("https://"):
        print("[MICROLM] ERRO: RIFT_RESULTS_ENDPOINT precisa usar HTTPS.")
        raise SystemExit(2)
    if not REPO_BASE.lower().startswith("https://"):
        print("[MICROLM] ERRO: o download da bateria precisa usar HTTPS.")
        raise SystemExit(2)

def enforce_publish_settings():
    if PUBLISH_MODE == "off":
        return
    token = os.environ.get("RIFT_INGEST_TOKEN", "").strip()
    if not token:
        print("[MICROLM] ERRO: RIFT_INGEST_TOKEN não chegou ao kernel do Colab.")
        print("[MICROLM] Cadastre o Secret RIFT_INGEST_TOKEN no Colab e rode a célula novamente.")
        raise SystemExit(2)
    if len(token) < 32:
        print("[MICROLM] ERRO: RIFT_INGEST_TOKEN precisa ter pelo menos 32 caracteres.")
        raise SystemExit(2)

def preclean_workspace():
    # Limpeza destrutiva só no Colab e só sob /content (contrato §5).
    if not os.path.isdir("/content"):
        return
    for path in PRECLEAN_PATHS:
        if not str(path).startswith("/content/"):
            continue
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
            print("[MICROLM] limpeza prévia:", path)

ROOT = Path("/content/microlm_run") if os.path.isdir("/content") else Path.cwd() / "microlm_run"

def download(repo_path, local_path):
    # Par (caminho_no_repo -> caminho_local) do contrato §20: a bateria e o
    # model.py caem juntos no diretório de execução (import do mesmo diretório).
    url = REPO_BASE + "/" + repo_path
    destination = ROOT / local_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "rift-microlm-launcher/1.0"})
    destination.write_bytes(urlopen(request, timeout=60).read())
    print("[MICROLM] download", repo_path, "->", local_path)

import_colab_secrets()
enforce_transport_security()
enforce_publish_settings()
preclean_workspace()
os.environ.setdefault("RIFT_RESULTS_ENDPOINT", RESULTS_ENDPOINT)
os.environ.setdefault("RIFT_GITHUB_REPOSITORY", GITHUB_REPOSITORY)
os.environ.setdefault("RIFT_SOURCE_REF", SOURCE_REF)
os.environ["RIFT_BENCHMARK_PROTOCOL"] = BENCHMARK_PROTOCOL
ROOT.mkdir(parents=True, exist_ok=True)
print("[MICROLM] Modelo de referência:", MODEL_ID, "| ref:", SOURCE_REF)
download(SCRIPT_REPO_PATH, SCRIPT_NAME)
for repo_path, local_path in MODEL_FILES:
    download(repo_path, local_path)
print("[MICROLM] executando", SCRIPT_NAME)
return_code = subprocess.call(
    [sys.executable, str(ROOT / SCRIPT_NAME), *ARGS],
    cwd=str(ROOT),
    env=os.environ.copy(),
)
print("[MICROLM] finalizado rc =", return_code)
if return_code != 0:
    raise SystemExit(return_code)
`;
}

/**
 * Conversor CASCADE (contrato §26.2 — CONVERTER_STATIC_V1): battery=converter
 * gera UMA célula Colab que importa os Secrets (HF_TOKEN de ESCRITA é
 * obrigatório quando hf_repo está presente; RIFT_INGEST_TOKEN é opcional e só
 * relevante com publish=on), instala as deps pinadas, baixa o runner
 * auto-contido de {origin}/converter.py e o executa com
 * --model <modelo> [--hf-repo <destino>] [--publish on]
 * --output /content/<nome>-cascade. A saída é o PRODUTO da conversão — por
 * isso ela NÃO entra na limpeza prévia; a nota de disco fica no ⓘ da célula
 * (comentários abaixo) e no card da UI (§26.3).
 */
function buildConverterLauncher({
  model,
  origin,
  hfRepo = null,
  publish = "off",
  keepSource = "off",
  repo = resolveRepo(),
  ref = resolveRef(),
}) {
  const compatibility = model.compatibility || { colabSupported: true };
  const modelName = model.modelId.split("/").pop();
  return `#!/usr/bin/env python3
# Célula do conversor CASCADE gerada por ${origin} (contrato §26.2 — CONVERTER_STATIC_V1).
# ⓘ Disco: a saída /content/${modelName}-cascade é o PRODUTO da conversão e NÃO é
# removida por limpeza prévia. O runner usa o modo pacote-por-pacote
# (--disk-budget-gb 75 + retomada) — garanta espaço livre no Colab (~120 GB de
# disco total) antes de converter modelos grandes.
# ⓘ Upload: com hf_repo o resultado sobe para o SEU repositório no Hugging Face
# (público/privado conforme a sua conta) usando o Secret HF_TOKEN de ESCRITA.
# Nenhum token é impresso ou embutido nesta célula.
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.request import Request, urlopen

ORIGIN = ${JSON.stringify(origin)}
RUNNER_URL = ORIGIN + ${JSON.stringify(CONVERTER_RUNNER_ROUTE)}
RESULTS_ENDPOINT = ${JSON.stringify(RESULTS_ENDPOINT)}
BENCHMARK_PROTOCOL = ${JSON.stringify(CONVERTER_BENCHMARK_PROTOCOL)}
GITHUB_REPOSITORY = ${JSON.stringify(repo)}
SOURCE_REF = ${JSON.stringify(ref)}
MODEL_ID = ${JSON.stringify(model.modelId)}
HF_REPO = ${hfRepo === null ? "None" : JSON.stringify(hfRepo)}
PUBLISH_MODE = ${JSON.stringify(publish)}
KEEP_SOURCE_MODE = ${JSON.stringify(keepSource)}
PIP_PACKAGES = json.loads(r'''${JSON.stringify(CONVERTER_PIP_PACKAGES)}''')
COMPATIBILITY = json.loads(r'''${JSON.stringify(compatibility)}''')
RUNNER_LOCAL_PATH = ${JSON.stringify(CONVERTER_RUNNER_LOCAL_PATH)}

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

def enforce_compatibility():
    if COMPATIBILITY.get("colabSupported", True):
        return
    print("[CONVERTER] BLOQUEADO: este modelo não é compatível com o Colab atual.")
    print("[CONVERTER] Motivo:", COMPATIBILITY.get("reason", "Recursos insuficientes."))
    print("[CONVERTER] Nenhum peso foi baixado e nada foi convertido.")
    raise SystemExit(2)

def enforce_transport_security():
    local = "://127.0.0.1" in RUNNER_URL or "://localhost" in RUNNER_URL
    if not RUNNER_URL.lower().startswith("https://") and not local:
        print("[CONVERTER] ERRO: o download do runner precisa usar HTTPS.")
        raise SystemExit(2)

def enforce_hf_token():
    # HF_TOKEN é obrigatório SOMENTE quando há repo de destino (upload).
    if not HF_REPO:
        return
    token = (os.environ.get("HF_TOKEN") or "").strip()
    if not token:
        print("[CONVERTER] ERRO: o upload para", HF_REPO, "exige o Secret HF_TOKEN")
        print("[CONVERTER] com permissão de ESCRITA (write) no Hugging Face.")
        print("[CONVERTER] Crie um token em huggingface.co/settings/tokens, cadastre-o")
        print("[CONVERTER] como Secret HF_TOKEN no Colab e rode a célula novamente.")
        raise SystemExit(2)

def report_ingest_token():
    # RIFT_INGEST_TOKEN é OPCIONAL: sem ele o conversor apenas recusa o publish
    # do dashboard, sem afetar a conversão nem o upload ao HF.
    if PUBLISH_MODE != "on":
        return
    token = (os.environ.get("RIFT_INGEST_TOKEN") or "").strip()
    if not token or len(token) < 32:
        print("[CONVERTER] AVISO: publish=on sem RIFT_INGEST_TOKEN válido (32+ chars);")
        print("[CONVERTER] o registro do dashboard será recusado, mas a conversão segue.")

def pip_install():
    print("[CONVERTER] instalando dependências pinadas:", ", ".join(PIP_PACKAGES))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *PIP_PACKAGES])

import_colab_secrets()
enforce_compatibility()
enforce_transport_security()
enforce_hf_token()
report_ingest_token()
os.environ.setdefault("RIFT_RESULTS_ENDPOINT", RESULTS_ENDPOINT)
os.environ.setdefault("RIFT_GITHUB_REPOSITORY", GITHUB_REPOSITORY)
os.environ.setdefault("RIFT_SOURCE_REF", SOURCE_REF)
os.environ["RIFT_BENCHMARK_PROTOCOL"] = BENCHMARK_PROTOCOL
pip_install()
runner_path = Path(RUNNER_LOCAL_PATH) if os.path.isdir("/content") else Path.cwd() / "cascade_converter_runner.py"
output_dir = ("/content/" if os.path.isdir("/content") else "") + MODEL_ID.split("/")[-1] + "-cascade"
print("[CONVERTER] baixando o runner:", RUNNER_URL, "->", str(runner_path))
request = Request(RUNNER_URL, headers={"User-Agent": "rift-converter-cell/1.0"})
runner_path.write_bytes(urlopen(request, timeout=60).read())
ARGS = ["--model", MODEL_ID, "--output", output_dir]
if HF_REPO:
    ARGS += ["--hf-repo", HF_REPO]
if PUBLISH_MODE == "on":
    ARGS += ["--publish", "on"]
if KEEP_SOURCE_MODE == "on":
    # Não copia os tensores que ficam fora do CASCADE (embeddings, lm_head,
    # MoE): eles seguem apontando para o checkpoint baixado. Evita o pico de
    # RAM/disco da cópia em modelo grande — o bundle passa a DEPENDER do
    # checkpoint, que por isso não pode ser apagado depois.
    ARGS += ["--keep-source-passthrough"]
print("[CONVERTER] executando o runner:", " ".join([str(runner_path), *ARGS]))
return_code = subprocess.call(
    [sys.executable, str(runner_path), *ARGS],
    env=os.environ.copy(),
)
print("[CONVERTER] finalizado rc =", return_code)
if return_code != 0:
    raise SystemExit(return_code)
print("[CONVERTER] saída CASCADE-DIR:", output_dir)
`;
}

function requestParameters(request) {
  const url = new URL(request.url);
  const battery = normalizeBattery(url.searchParams.get("battery"));
  if (battery === "cap") {
    // Bateria de capacidades (§9): avalia o modelo baseline, sem tecnologia
    // de otimização — technology/target_layer/device/arch não se aplicam.
    return {
      technology: null,
      model: normalizeModel(url.searchParams.get("model")),
      publish: normalizePublish(url.searchParams.get("publish")),
      battery,
      origin: url.origin,
    };
  }
  if (battery === "microlm") {
    // MicroLM (§22): o MicroLM É o modelo (referência fixa
    // microlm/MicroLM-22M-v0.2) — a rota /microlm não tem segmento de modelo
    // e a AUSÊNCIA do parâmetro é o caminho válido. Um model explícito é
    // rejeitado com mensagem clara (nunca o genérico "Modelo inválido").
    const rawModel = String(url.searchParams.get("model") || "").trim();
    if (rawModel) {
      throw new ApiError(
        "battery=microlm não aceita parâmetro de modelo: a bateria avalia o "
        + "modelo de referência fixo microlm/MicroLM-22M-v0.2 (contrato §22). "
        + "Use GET /microlm sem modelo.",
      );
    }
    return {
      technology: null,
      model: MICROLM_MODEL,
      publish: normalizePublish(url.searchParams.get("publish")),
      battery,
      origin: url.origin,
    };
  }
  if (battery === "converter") {
    // Conversor CASCADE (§26.2): não é uma bateria de benchmark de tecnologia —
    // technology/target_layer/device/arch não se aplicam. Parâmetros próprios:
    // hf_repo (opcional, org/nome validado — 400 se inválido), publish on|off e
    // keep_source on|off (§29.3).
    return {
      technology: null,
      model: normalizeModel(url.searchParams.get("model")),
      hfRepo: normalizeHfRepo(url.searchParams.get("hf_repo")),
      publish: normalizeConverterPublish(url.searchParams.get("publish")),
      keepSource: normalizeConverterKeepSource(url.searchParams.get("keep_source")),
      battery,
      origin: url.origin,
    };
  }
  if (battery === "gguf") {
    // Caminho GGUF (§11): o runtime é o llama.cpp — technology/target_layer/
    // device/arch não se aplicam; a trava anti-GGUF/anti-NVFP4 é dispensada.
    return {
      technology: null,
      model: ggufRuntimeModel(normalizeModel(url.searchParams.get("model"))),
      quant: normalizeQuant(url.searchParams.get("quant")),
      publish: normalizePublish(url.searchParams.get("publish")),
      battery,
      origin: url.origin,
    };
  }
  if (battery === "c3") {
    // Série C3 (§2): technology=all dispara a fila serial completa (§13.3);
    // as quatro tecnologias individuais seguem aceitas para automação.
    const rawTechnology = String(url.searchParams.get("technology") || "").trim().toLowerCase();
    const technology = rawTechnology === "all" ? "all" : normalizeTechnology(rawTechnology);
    if (technology !== "all" && !C3_TECHNOLOGIES.has(technology)) {
      throw new ApiError(
        "A série C3 aceita apenas rift, aether, cascade ou spectra "
        + "(ou all para a fila serial completa)",
      );
    }
    return {
      technology,
      model: normalizeModel(url.searchParams.get("model")),
      publish: normalizePublish(url.searchParams.get("publish")),
      battery,
      origin: url.origin,
    };
  }
  if (battery === "final") {
    // Fase final (§16): technology=all roda as 4 tecnologias em fila serial;
    // as rotas por tecnologia seguem aceitas para automação.
    const rawTechnology = String(url.searchParams.get("technology") || "").trim().toLowerCase();
    const technology = rawTechnology === "all" ? "all" : normalizeTechnology(rawTechnology);
    if (technology !== "all" && !FINAL_TECHNOLOGIES.has(technology)) {
      throw new ApiError(
        "A fase final aceita apenas rift, aether, cascade ou spectra "
        + "(ou all para a fila serial completa)",
      );
    }
    return {
      technology,
      model: normalizeModel(url.searchParams.get("model")),
      publish: normalizePublish(url.searchParams.get("publish")),
      battery,
      origin: url.origin,
    };
  }
  const technology = normalizeTechnology(url.searchParams.get("technology"));
  const model = normalizeModel(url.searchParams.get("model"));
  const targetLayer = normalizeTargetLayer(url.searchParams.get("target_layer"));
  const device = normalizeDevice(url.searchParams.get("device"));
  const publish = normalizePublish(url.searchParams.get("publish"));
  const winnerArch = normalizeWinnerArch(url.searchParams.get("arch"));
  const trustRemoteCode = ["1", "true", "yes"].includes(
    String(url.searchParams.get("trust_remote_code") || "").toLowerCase(),
  );
  return {
    technology,
    model,
    targetLayer,
    device,
    publish,
    battery,
    winnerArch,
    trustRemoteCode,
    origin: url.origin,
  };
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
      const modelSlug = parameters.model.modelId.replace("/", "-");
      let body;
      let filename;
      if (parameters.battery === "c3") {
        body = buildC3Launcher(parameters);
        filename = `c3-${parameters.technology}-${modelSlug}.py`;
      } else if (parameters.battery === "final") {
        body = buildFinalLauncher(parameters);
        filename = `final-${parameters.technology}-${modelSlug}.py`;
      } else if (parameters.battery === "cap") {
        body = buildCapLauncher(parameters);
        filename = `cap-${modelSlug}.py`;
      } else if (parameters.battery === "gguf") {
        body = buildGgufLauncher(parameters);
        filename = `gguf-${modelSlug}-${parameters.quant}.py`;
      } else if (parameters.battery === "microlm") {
        body = buildMicrolmLauncher(parameters);
        filename = "microlm-m0.py";
      } else if (parameters.battery === "converter") {
        body = buildConverterLauncher(parameters);
        filename = `converter-${modelSlug}.py`;
      } else {
        body = buildLauncher(parameters);
        filename = `${parameters.technology}-${modelSlug}.py`;
      }
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
  C3_ALL_TECHNOLOGIES,
  C3_BENCHMARK_PROTOCOL,
  C3_PACKAGE_FILES,
  C3_SCRIPT,
  C3_SCRIPT_REPO_PATH,
  CAP_BENCHMARK_PROTOCOL,
  CAP_SCRIPT,
  CAP_SCRIPT_REPO_PATH,
  COLAB_PRECLEAN_PATHS,
  CONVERTER_BENCHMARK_PROTOCOL,
  CONVERTER_PIP_PACKAGES,
  CONVERTER_RUNNER_LOCAL_PATH,
  CONVERTER_RUNNER_ROUTE,
  FINAL_ALL_TECHNOLOGIES,
  FINAL_BENCHMARK_PROTOCOL,
  FINAL_SCRIPT,
  FINAL_SCRIPT_REPO_PATH,
  GGUF_BENCHMARK_PROTOCOL,
  GGUF_DEFAULT_QUANT,
  GGUF_OUTPUT_DIR,
  GGUF_SCRIPT,
  GGUF_SCRIPT_REPO_PATH,
  MICROLM_BENCHMARK_PROTOCOL,
  MICROLM_MODEL_FILES,
  MICROLM_MODEL_ID,
  MICROLM_SCRIPT,
  MICROLM_SCRIPT_REPO_PATH,
  HF_REPO_RE,
  buildC3AllLauncher,
  buildC3Launcher,
  buildCapLauncher,
  buildConverterLauncher,
  buildFinalLauncher,
  buildGgufLauncher,
  buildLauncher,
  buildMicrolmLauncher,
  ggufRuntimeModel,
  normalizeBattery,
  normalizeConverterKeepSource,
  normalizeConverterPublish,
  normalizeDevice,
  normalizeHfRepo,
  normalizeModel,
  normalizePublish,
  normalizeQuant,
  normalizeTargetLayer,
  normalizeTechnology,
  normalizeWinnerArch,
  requestParameters,
};
