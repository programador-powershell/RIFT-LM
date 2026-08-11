#!/usr/bin/env python3
"""WINNER.cpp v0.8 native self-test + real-tensor reference battery for Colab.

The native C++ suite validates the reviewed runtime. The primary dashboard
battery separately measures F0 + low-rank residual on a real Hugging Face
Linear tensor with PyTorch. The two latency scopes are never conflated.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


torch = None
F = None
AutoModel = None
AutoModelForCausalLM = None
AutoTokenizer = None
AutoModelForMultimodalLM = None

# Repo-agnóstico (docs/C3_CONTRACTS_V1.md §14.2): owner/repo e ref vêm do
# ambiente (RIFT_GITHUB_REPOSITORY / RIFT_SOURCE_REF, exportados pelas células
# geradas no servidor); o literal legado é APENAS o fallback documentado.
LEGACY_REPOSITORY = "programador-powershell/RIFT-LM"
REPOSITORY_ARCHIVE_TEMPLATE = "https://github.com/{repository}/archive/{ref}.tar.gz"
MAX_ARCHIVE_BYTES = 25 * 1024 * 1024

BENCHMARK_PROTOCOL = "LINEAR_REFERENCE_V2"
DEFAULT_RESULTS_ENDPOINT = "https://rift-lm.vercel.app/api/results"
HISTORY_FALLBACK_TEMPLATE = "https://raw.githubusercontent.com/{repository}/{ref}/data/rift_test_batteries.json"

# Bateria E2E de tok/s (docs/C3_CONTRACTS_V1.md §12): prompt PT-BR fixo + greedy.
E2E_GENERATION_PROMPT = "Liste três técnicas para reduzir o uso de memória na inferência de LLMs:"
E2E_MAX_NEW_TOKENS = 48
E2E_MAX_PARAMS = 3_000_000_000  # mesmo guard do G3_GEYSER_BURST: acima disso → SKIPPED

# WINNER dinâmico (docs/C3_CONTRACTS_V1.md §1): o próprio WINNER é excluído.
# CAP (bateria de capacidades, §9) NUNCA é elegível — registros technology="CAP"
# são ignorados por não constarem no conjunto abaixo.
WINNER_ELIGIBLE_TECHNOLOGIES = ("RIFT", "AETHER", "CASCADE", "SPECTRA", "GEYSER")
WINNER_PRIORITY_ORDER = ("CASCADE", "RIFT", "AETHER", "SPECTRA", "GEYSER")
# Espelha SCORE_WEIGHTS de api/analyze.mjs / index.html (implementações espelhadas).
# Score canônico v2 (docs/C3_CONTRACTS_V1.md §25): melhor tecnologia para PC
# convencional (4 núcleos, 8 GB de RAM livre, sem GPU) — RAM é a restrição dura.
# Qualidade 40% • RAM 30% • latência/velocidade 20% • disco 10%.
# Normalizações do §1 e fator de coverage NÃO mudam; apenas os pesos.
SCORE_WEIGHTS = {
    "output_cosine": 25,
    "output_nrmse": 10,
    "disk_reduction_pct": 10,
    "ram_reduction_pct": 30,
    "operation_speedup_x": 20,
    "quality_gate_pass": 5,
}

# pip automático: somente Colab ou RIFT_AUTO_INSTALL=1; versões com piso+teto testados
# (nunca -U para "latest" sem pino) — docs/C3_CONTRACTS_V1.md §5.
PIP_PINNED_SPECS = {
    "sentencepiece": "sentencepiece>=0.1.99,<0.3",
    "tiktoken": "tiktoken>=0.7,<1",
    "transformers": "transformers>=4.44,<5",
    "accelerate": "accelerate>=0.33,<2",
    "huggingface_hub": "huggingface_hub>=0.24,<1",
}


class ResultsPublishError(RuntimeError):
    pass


def running_in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


def pip_auto_install_allowed() -> bool:
    """pip automático só no Colab ou com RIFT_AUTO_INSTALL=1 (contrato §5)."""
    if os.environ.get("RIFT_AUTO_INSTALL", "").strip() == "1":
        return True
    return running_in_colab()


def package_version(name: str) -> str | None:
    try:
        from importlib import metadata
        return metadata.version(name)
    except Exception:
        pass
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "") or "") or None
    except Exception:
        return None


def ensure_import(module: str, pip_name: str | None = None):
    try:
        return __import__(module)
    except ImportError:
        package = pip_name or module
        spec = PIP_PINNED_SPECS.get(package, package)
        if not pip_auto_install_allowed():
            raise SystemExit(
                f"Dependência ausente: {package}. Instalação automática desativada fora do Colab. "
                f"Instale manualmente (pip install \"{spec}\") ou defina RIFT_AUTO_INSTALL=1."
            )
        print(f"[deps] Instalando {spec}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", spec])
        return __import__(module)


def resolve_source_repository() -> str:
    """owner/repo de RIFT_GITHUB_REPOSITORY com fallback legado (contrato §14.2).

    Mesma validação de owner/repo usada na publicação GitHub do script RIFT
    (rift_m0_phase1_test_v035_auto_batteries.py::_normalize_github_repository).
    """
    candidate = os.environ.get("RIFT_GITHUB_REPOSITORY", "").strip().rstrip("/")
    if not candidate:
        return LEGACY_REPOSITORY
    if candidate.lower().endswith(".git"):
        candidate = candidate[:-4]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", candidate):
        raise RuntimeError(
            "RIFT_GITHUB_REPOSITORY inválido. Use o formato 'owner/repo'."
        )
    return candidate


def resolve_source_ref() -> str:
    """Ref de RIFT_SOURCE_REF (branch, tag ou SHA; seguro para URL); default 'main'.

    Aceita SHA de 40 hex (pin preferido, contrato §14.1) ou nomes de branch/tag
    simples; rejeita qualquer coisa fora de [A-Za-z0-9._/-] ou com '..'.
    """
    ref = os.environ.get("RIFT_SOURCE_REF", "").strip() or "main"
    if ".." in ref or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,199}", ref):
        raise RuntimeError("RIFT_SOURCE_REF inválido")
    return ref


def repository_archive() -> str:
    return REPOSITORY_ARCHIVE_TEMPLATE.format(
        repository=resolve_source_repository(), ref=resolve_source_ref()
    )


def history_fallback_url() -> str:
    """URL raw do histórico publicado (único ponto que monta raw.githubusercontent).

    O snapshot data/rift_test_batteries.json vive na ponta da branch, então a
    ref preferida é RIFT_GITHUB_BRANCH; sem ela, usa RIFT_SOURCE_REF quando for
    branch/tag (um SHA pinado apontaria para histórico congelado) e cai em
    'main' caso contrário.
    """
    branch = os.environ.get("RIFT_GITHUB_BRANCH", "").strip()
    if not branch:
        source_ref = resolve_source_ref()
        branch = "main" if re.fullmatch(r"[a-fA-F0-9]{40}", source_ref) else source_ref
    if ".." in branch or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,199}", branch):
        raise RuntimeError("RIFT_GITHUB_BRANCH inválido")
    return HISTORY_FALLBACK_TEMPLATE.format(
        repository=resolve_source_repository(), ref=branch
    )


def ensure_ml_dependencies() -> None:
    global torch, F, AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoModelForMultimodalLM
    if torch is not None:
        return
    if pip_auto_install_allowed():
        ensure_import("sentencepiece")
        ensure_import("tiktoken")
        # Gemma 4 Unified / modelos recentes exigem Transformers na faixa testada (pinada)
        print("[deps] Garantindo transformers/accelerate/huggingface_hub na faixa testada (Gemma 4 / multimodal)...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            PIP_PINNED_SPECS["transformers"], PIP_PINNED_SPECS["accelerate"], PIP_PINNED_SPECS["huggingface_hub"],
        ])
    else:
        print("[deps] Instalação automática desativada (sem google.colab e sem RIFT_AUTO_INSTALL=1); usando pacotes já instalados.")
        for optional in ("sentencepiece", "tiktoken"):
            try:
                __import__(optional)
            except ImportError:
                print(f"[deps] AVISO: {optional} indisponível; alguns tokenizers podem falhar.")
    try:
        import torch as _torch
        import torch.nn.functional as _F
        from transformers import AutoModel as _AutoModel
        from transformers import AutoModelForCausalLM as _AutoModelForCausalLM
        from transformers import AutoTokenizer as _AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "PyTorch e Transformers são necessários. Instale com: "
            "pip install torch transformers accelerate sentencepiece tiktoken\n"
            f"Erro: {exc}"
        ) from exc
    _AutoMM = None
    try:
        from transformers import AutoModelForMultimodalLM as _AutoMM
    except ImportError:
        try:
            from transformers import AutoModelForImageTextToText as _AutoMM
        except ImportError:
            _AutoMM = None
    torch = _torch
    F = _F
    AutoModel = _AutoModel
    AutoModelForCausalLM = _AutoModelForCausalLM
    AutoTokenizer = _AutoTokenizer
    AutoModelForMultimodalLM = _AutoMM


def normalize_huggingface_model_id(value: str) -> str:
    raw = str(value or "").strip()
    parsed = urlparse(raw)
    if parsed.scheme:
        if parsed.scheme != "https" or parsed.netloc.lower() not in {
            "huggingface.co", "www.huggingface.co",
        }:
            raise ValueError("Use um model ID org/modelo ou URL HTTPS do huggingface.co")
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            raise ValueError("URL do Hugging Face precisa conter org/modelo")
        raw = f"{parts[0]}/{parts[1]}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", raw):
        raise ValueError("Model ID inválido; use org/modelo")
    return raw


def resolve_linear_weight_name(model: Any, requested: str) -> str:
    state = model.state_dict()
    if requested != "auto":
        name = requested if requested.endswith(".weight") else f"{requested}.weight"
        tensor = state.get(name)
        if tensor is None or getattr(tensor, "ndim", 0) != 2:
            raise ValueError(f"Tensor Linear 2D não encontrado: {name}")
        return name
    preferred = (
        # Gemma 4 Unified (language_model backbone)
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.mlp.down_proj.weight",
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "language_model.layers.0.self_attn.q_proj.weight",
        "language_model.layers.0.mlp.down_proj.weight",
        # Gemma / Llama / Qwen style
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.model.layers.0.self_attn.q_proj.weight",
        "model.model.layers.0.mlp.down_proj.weight",
        # GPT-2 / OPT style
        "transformer.h.0.attn.c_attn.weight",
        "transformer.h.0.mlp.c_proj.weight",
        # Phi / others
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.layers.0.mixer.Wqkv.weight",
    )
    for name in preferred:
        if name in state and state[name].ndim == 2:
            return name
    for name, tensor in state.items():
        if name.endswith(".weight") and getattr(tensor, "ndim", 0) == 2:
            if min(tensor.shape) >= 32:
                return name
    raise ValueError("Nenhum tensor Linear 2D compatível foi encontrado")


def capture_activation(model: Any, tokenizer: Any, module_name: str, device: Any, prompt: str):
    modules = dict(model.named_modules())
    module = modules.get(module_name)
    if module is None:
        raise ValueError(f"Módulo não encontrado para capturar ativação: {module_name}")
    captured: list[Any] = []

    def hook(_module, inputs):
        if inputs and hasattr(inputs[0], "detach"):
            captured.append(inputs[0].detach())

    handle = module.register_forward_pre_hook(hook)
    try:
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            model(**encoded)
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError("O forward não produziu ativação na camada selecionada")
    x = captured[0].reshape(-1, captured[0].shape[-1]).to(device=device, dtype=torch.float32)
    return x[-min(32, x.shape[0]):]





def resolve_hf_token() -> str | None:
    """HF_TOKEN / HUGGING_FACE_HUB_TOKEN a partir do ambiente ou Secrets do Colab."""
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    try:
        from google.colab import userdata
        for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
            try:
                value = str(userdata.get(name) or "").strip()
            except Exception:
                value = ""
            if value:
                # Espelha no ambiente para transformers/huggingface_hub
                os.environ.setdefault("HF_TOKEN", value)
                os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", value)
                return value
    except Exception:
        pass
    return None


def ensure_hf_login(token: str | None = None) -> str | None:
    """Autentica no Hugging Face Hub quando há token (modelos gated)."""
    token = token or resolve_hf_token()
    if not token:
        return None
    try:
        from huggingface_hub import login as hf_login
        hf_login(token=token, add_to_git_credential=False)
        print("[auth] HF_TOKEN aplicado (valor não exibido).")
    except Exception as exc:
        print(f"[auth] AVISO: não foi possível fazer login no Hub: {exc}")
    return token


def is_gated_access_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "gated repo",
        "access to model",
        "restricted",
        "401 client error",
        "403 client error",
        "cannot access gated",
        "you must have access",
        "please log in",
        "authentication",
        "authorized",
    )
    return any(m in text for m in markers)


def resolve_torch_device(requested: str):
    """cuda se disponível; senão CPU (Colab sem GPU / TPU sem CUDA)."""
    requested = (requested or "auto").strip().lower()
    if requested in {"auto", "gpu"}:
        requested = "cuda"
    if requested not in {"cuda", "cpu"}:
        raise ValueError(f"device inválido: {requested} (use auto, cuda ou cpu)")
    if requested == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0) if torch.cuda.device_count() else "CUDA"
                print(f"[device] CUDA disponível → usando GPU ({name})")
                return torch.device("cuda")
        except Exception as exc:
            print(f"[device] CUDA indisponível ({exc}); caindo para CPU")
        print("[device] Sem GPU CUDA — executando em CPU (adequado a Colab CPU/TPU sem torch_xla)")
        return torch.device("cpu")
    print("[device] Forçado para CPU")
    return torch.device("cpu")



def _cleanup_target_allowed(path: Path, *, colab: bool, allow_local: bool) -> bool:
    """Limpeza destrutiva (contrato §5): só no Colab, com RIFT_ALLOW_LOCAL_CLEANUP=1
    ou quando o alvo está sob /content ou /tmp. Nunca apaga diretórios locais
    (HOME/cwd) em máquina fora do Colab sem opt-in explícito."""
    if colab or allow_local:
        return True
    text = str(path).replace("\\", "/")
    return text.startswith("/content/") or text.startswith("/tmp/") or text in {"/content", "/tmp"}


def cleanup_colab_workspace(*, label: str = "battery", wipe_hf_cache: bool = False) -> None:
    """Libera artefatos temporários no Colab.

    Por padrão NÃO apaga o cache Hugging Face entre tecnologias da mesma célula
    serial (evita re-download de dezenas de GB). Wipe completo do hub só com
    wipe_hf_cache=True (final da fila / célula).
    """
    import gc
    import shutil
    import glob as _glob

    colab = running_in_colab()
    allow_local = os.environ.get("RIFT_ALLOW_LOCAL_CLEANUP", "").strip() == "1"
    skipped = 0

    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass

    removed = []
    if wipe_hf_cache:
        home = Path.home()
        targets = [
            home / ".cache" / "huggingface" / "hub",
            home / ".cache" / "huggingface" / "transformers",
            home / ".cache" / "huggingface" / "modules",
            home / ".cache" / "torch",
            Path("/content") / ".cache",
            Path("/root") / ".cache" / "huggingface" / "hub",
            Path("/root") / ".cache" / "huggingface" / "transformers",
        ]
        for path in targets:
            if not _cleanup_target_allowed(path, colab=colab, allow_local=allow_local):
                skipped += 1
                continue
            try:
                if path.is_dir():
                    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                    shutil.rmtree(path, ignore_errors=True)
                    removed.append(f"{path} (~{size / (1024**3):.2f} GiB)")
                elif path.is_file():
                    path.unlink(missing_ok=True)
                    removed.append(str(path))
            except Exception as exc:
                print(f"[cleanup] AVISO ao remover {path}: {exc}")

    patterns = [
        "/tmp/winner_cpp_*",
        "/tmp/winner_phase1_*",
        "/tmp/phase1_load_fail*",
        "/tmp/cascade_load_fail*",
        "/tmp/rift_*",
        "/content/*_launcher.py",
            ]
    for pattern in patterns:
        for match in _glob.glob(pattern):
            p = Path(match)
            if not _cleanup_target_allowed(p, colab=colab, allow_local=allow_local):
                skipped += 1
                continue
            try:
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                elif p.is_file():
                    p.unlink(missing_ok=True)
                removed.append(str(p))
            except Exception as exc:
                print(f"[cleanup] AVISO ao remover {p}: {exc}")

    if skipped:
        print(
            f"[cleanup] {label}: {skipped} alvo(s) fora de /content|/tmp ignorado(s) em execução local "
            "(defina RIFT_ALLOW_LOCAL_CLEANUP=1 para forçar)"
        )
    if removed:
        print(f"[cleanup] {label}: espaço liberado ({len(removed)} item(ns)):")
        for item in removed[:12]:
            print(f"  - {item}")
        if len(removed) > 12:
            print(f"  - … +{len(removed) - 12} outros")
    else:
        print(f"[cleanup] {label}: nada temporário para limpar (cache HF preservado)")
    gc.collect()




def load_tokenizer(model_id: str, *, trust_remote_code: bool = False, token: str | None = None):
    """Carrega tokenizer com fallbacks (subfolder, use_fast=False).

    Para de tentar subpastas se o Hub responder 401/gated — o problema é auth,
    não layout de arquivos.
    """
    token = token or resolve_hf_token()
    ensure_hf_login(token)
    common = {"trust_remote_code": trust_remote_code}
    if token:
        common["token"] = token
    attempts = [
        {},
        {"use_fast": False},
        {"subfolder": "tokenizer"},
        {"subfolder": "tokenizer", "use_fast": False},
        {"subfolder": "processor"},
        {"subfolder": "processor", "use_fast": False},
    ]
    errors: list[str] = []
    for extra in attempts:
        try:
            tok = AutoTokenizer.from_pretrained(model_id, **common, **extra)
            print(f"[tokenizer] OK com {extra or {'root': True}}")
            return tok
        except Exception as exc:
            errors.append(f"{extra or 'root'}: {type(exc).__name__}: {exc}")
            if is_gated_access_error(exc):
                raise RuntimeError(
                    f"Modelo gated/restrito: {model_id}.\n"
                    "1) Aceite os termos em https://huggingface.co/" + model_id + "\n"
                    "2) Configure o Secret HF_TOKEN no Colab (token com acesso de leitura).\n"
                    "3) Rode de novo. Sem token válido a bateria grava FAIL e segue a fila."
                ) from exc
    raise RuntimeError(
        "Não foi possível carregar o tokenizer de "
        f"{model_id}.\nTentativas:\n- " + "\n- ".join(errors)
    )



def load_model(model_id: str, *, device: Any, trust_remote_code: bool):
    token = ensure_hf_login(resolve_hf_token())
    common = {"trust_remote_code": trust_remote_code, "token": token}
    tokenizer = load_tokenizer(model_id, trust_remote_code=trust_remote_code, token=token)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    load_kwargs = {
        **common,
        "low_cpu_mem_usage": True,
        "dtype": dtype,
    }
    # Modelos grandes (Gemma 4 12B ~24GB) precisam de device_map para caber em Colab
    if device.type == "cuda":
        load_kwargs["device_map"] = "auto"
    classes = []
    if AutoModelForMultimodalLM is not None:
        classes.append(AutoModelForMultimodalLM)
    classes.extend([AutoModelForCausalLM, AutoModel])
    errors = []
    for cls in classes:
        try:
            model = cls.from_pretrained(model_id, **load_kwargs)
            if getattr(model, "hf_device_map", None) is None:
                model = model.to(device)
            model.eval()
            print(f"[load] Modelo carregado via {cls.__name__}")
            return tokenizer, model
        except Exception as exc:
            errors.append(f"{cls.__name__}: {exc}")
    raise RuntimeError(
        "Não foi possível carregar o modelo (tokenizer OK).\n"
        "Gemma 4 Unified exige transformers recente e costuma usar "
        "AutoModelForMultimodalLM. Modelos diffusers/vídeo puros não se aplicam.\n"
        + "\n".join(errors)
    )



def compute_metrics(reference: Any, candidate: Any) -> dict[str, float]:
    ref = reference.float().reshape(-1)
    pred = candidate.float().reshape(-1)
    cosine = float(F.cosine_similarity(ref, pred, dim=0).item())
    error = torch.linalg.vector_norm(ref - pred)
    denominator = torch.linalg.vector_norm(ref).clamp_min(1e-12)
    return {"cosine": cosine, "nrmse": float((error / denominator).item())}


def benchmark_ms(fn, *, device: Any, iterations: int) -> dict[str, float]:
    for _ in range(3):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    values = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        values.append((time.perf_counter_ns() - start) / 1_000_000)
    return {"median_ms": statistics.median(values), "min_ms": min(values), "iterations": iterations}


def _read_vmrss_bytes() -> int | None:
    """VmRSS de /proc/self/status em bytes (Linux/Colab); None fora do Linux."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        return None
    return None


def _read_meminfo_available_bytes() -> int | None:
    """MemAvailable via /proc/meminfo (Linux/Colab); None fora do Linux."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


def _getrusage_peak_bytes() -> int | None:
    """Fallback: pico ru_maxrss do processo (KB no Linux)."""
    try:
        import resource
        peak_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return peak_kb * 1024 if peak_kb > 0 else None
    except Exception:
        return None


class VmRssSampler:
    """Thread que amostra VmRSS (~1ms) durante uma fase de benchmark (contrato §3)."""

    def __init__(self, interval_s: float = 0.001):
        self._interval_s = interval_s
        self._samples: list[int] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        def _loop() -> None:
            while not self._stop_event.is_set():
                value = _read_vmrss_bytes()
                if value is not None:
                    self._samples.append(value)
                self._stop_event.wait(self._interval_s)

        self._thread = threading.Thread(target=_loop, name="vmrss-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        samples = self._samples
        if not samples:
            return {"method": "proc_vmrss_sampling_per_phase_v1", "max_bytes": None, "mean_bytes": None, "samples": 0}
        return {
            "method": "proc_vmrss_sampling_per_phase_v1",
            "max_bytes": int(max(samples)),
            "mean_bytes": int(sum(samples) / len(samples)),
            "samples": len(samples),
        }


def benchmark_ms_with_ram(fn, *, device: Any, iterations: int) -> tuple[dict[str, float], dict[str, Any]]:
    """benchmark_ms + RAM real da fase (VmRSS ~1ms; fallback getrusage; senão None)."""
    if _read_vmrss_bytes() is not None:
        sampler = VmRssSampler()
        sampler.start()
        try:
            perf = benchmark_ms(fn, device=device, iterations=iterations)
        finally:
            ram = sampler.stop()
        return perf, ram
    perf = benchmark_ms(fn, device=device, iterations=iterations)
    peak = _getrusage_peak_bytes()
    if peak is not None:
        return perf, {"method": "getrusage_peak_fallback", "max_bytes": peak, "mean_bytes": None, "samples": None}
    return perf, {"method": None, "max_bytes": None, "mean_bytes": None, "samples": None}


def measured_phase_max(ram_phase: dict[str, Any] | None) -> int | None:
    """*_ram_bytes de nível superior: apenas VmRSS máximo medido por fase; senão None.

    getrusage é pico acumulado do processo (não por fase) e fica só em metrics.memory.
    """
    if not isinstance(ram_phase, dict):
        return None
    if ram_phase.get("method") != "proc_vmrss_sampling_per_phase_v1":
        return None
    value = ram_phase.get("max_bytes")
    return int(value) if isinstance(value, (int, float)) and value else None


def _pad_columns(matrix: Any, multiple: int):
    matrix = matrix.contiguous()
    rows, cols = matrix.shape
    padded_cols = ((cols + multiple - 1) // multiple) * multiple
    if padded_cols == cols:
        return matrix, cols
    out = torch.zeros((rows, padded_cols), dtype=matrix.dtype, device=matrix.device)
    out[:, :cols] = matrix
    return out, cols


def _tensor_payload(*tensors: Any) -> bytes:
    """Bytes REAIS dos buffers empacotados (para artefatos F0/F1 via os.stat)."""
    return b"".join(t.detach().cpu().contiguous().numpy().tobytes() for t in tensors)


def quantize_int4_groupwise(weight: Any, *, group_size: int = 32):
    """F0 do CASCADE: INT4 groupwise signed [-8,7], grupo 32, escalas FP16."""
    padded, cols = _pad_columns(weight.float(), group_size)
    rows = int(padded.shape[0])
    groups = padded.reshape(rows, -1, group_size)
    scales = (groups.abs().amax(dim=2) / 7.0).clamp_min(1e-12).to(torch.float16)
    scale_f32 = scales.float().unsqueeze(2)
    codes = torch.clamp(torch.round(groups / scale_f32), -8, 7).to(torch.int8)
    nibbles = (codes.to(torch.int16) + 8).to(torch.uint8).reshape(rows, -1)
    packed = (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)).contiguous()
    dequant = (codes.float() * scale_f32).reshape(rows, -1)[:, :cols].contiguous()
    codes_bytes = int(packed.numel() * packed.element_size())
    scales_bytes = int(scales.numel() * scales.element_size())
    meta = {
        "codec": "int4_groupwise_g32",
        "group_size": group_size,
        "levels": "signed[-8,7]",
        "scales_dtype": "float16",
        "codes_bytes": codes_bytes,
        "scales_bytes": scales_bytes,
    }
    return dequant, codes_bytes + scales_bytes, meta, _tensor_payload(packed, scales)


def quantize_int2_groupwise(weight: Any, *, group_size: int = 32, role: str = "base"):
    """F0/F1 do RIFT: 2 bits/peso groupwise, 4 níveis simétricos (±0.5/±1.5 × step), escalas FP16."""
    padded, cols = _pad_columns(weight.float(), group_size)
    rows = int(padded.shape[0])
    groups = padded.reshape(rows, -1, group_size)
    steps = (groups.abs().amax(dim=2) / 1.5).clamp_min(1e-12).to(torch.float16)
    step_f32 = steps.float().unsqueeze(2)
    codes = torch.clamp(torch.round(groups / step_f32 + 1.5), 0, 3).to(torch.uint8)
    flat = codes.reshape(rows, -1)
    packed = (flat[:, 0::4] | (flat[:, 1::4] << 2) | (flat[:, 2::4] << 4) | (flat[:, 3::4] << 6)).contiguous()
    dequant = ((codes.float() - 1.5) * step_f32).reshape(rows, -1)[:, :cols].contiguous()
    codes_bytes = int(packed.numel() * packed.element_size())
    scales_bytes = int(steps.numel() * steps.element_size())
    meta = {
        "codec": "int2_groupwise_g32",
        "kind": "int2_groupwise_refinement" if role == "refinement" else "int2_groupwise_base",
        "group_size": group_size,
        "levels": "symmetric4(+-0.5,+-1.5)xstep",
        "scales_dtype": "float16",
        "codes_bytes": codes_bytes,
        "scales_bytes": scales_bytes,
    }
    return dequant, codes_bytes + scales_bytes, meta, _tensor_payload(packed, steps)


def quantize_zdc_int2_asymmetric_groupwise(weight: Any, *, group_size: int = 64):
    """F0 do GEYSER: ZDC INT2g64 — 2 bits/peso groupwise ASSIMÉTRICO (espelha
    zdc_quant_int2 da spec GEYSER v0.1): níveis {0..3}, escala por grupo
    (max-min)/3 em FP16 + mínimo por grupo em FP16, 4 códigos por byte.
    Dequant: code * scale + min. Bytes reportados são dos buffers reais."""
    padded, cols = _pad_columns(weight.float(), group_size)
    rows = int(padded.shape[0])
    groups = padded.reshape(rows, -1, group_size)
    group_min = groups.amin(dim=2)
    group_max = groups.amax(dim=2)
    scales = ((group_max - group_min) / 3.0).clamp_min(1e-12).to(torch.float16)
    mins = group_min.to(torch.float16)
    scale_f32 = scales.float().unsqueeze(2)
    min_f32 = mins.float().unsqueeze(2)
    codes = torch.clamp(torch.round((groups - min_f32) / scale_f32), 0, 3).to(torch.uint8)
    flat = codes.reshape(rows, -1)
    packed = (flat[:, 0::4] | (flat[:, 1::4] << 2) | (flat[:, 2::4] << 4) | (flat[:, 3::4] << 6)).contiguous()
    dequant = (codes.float() * scale_f32 + min_f32).reshape(rows, -1)[:, :cols].contiguous()
    codes_bytes = int(packed.numel() * packed.element_size())
    scales_bytes = int(scales.numel() * scales.element_size()) + int(mins.numel() * mins.element_size())
    meta = {
        "codec": "zdc_int2_groupwise_g64",
        "group_size": group_size,
        "levels": "asymmetric{0..3}",
        "scales_dtype": "float16",
        "mins_dtype": "float16",
        "scale_rule": "(max-min)/3",
        "codes_bytes": codes_bytes,
        "scales_bytes": scales_bytes,
    }
    return dequant, codes_bytes + scales_bytes, meta, _tensor_payload(packed, scales, mins)


def quantize_ternary_rowscale(weight: Any):
    """F0 do AETHER/SPECTRA: ternário {-1,0,+1} com escala por linha (busca de limiar) — caminho existente."""
    row_scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    normalized = weight / row_scale
    best_error = None
    best_threshold = None
    best_codes = None
    for step in range(1, 19):
        threshold = step * 0.05
        codes = torch.where(
            normalized > threshold, torch.ones_like(weight),
            torch.where(normalized < -threshold, -torch.ones_like(weight), torch.zeros_like(weight)),
        )
        error = float(torch.linalg.vector_norm(weight - codes * row_scale).item())
        if best_error is None or error < best_error:
            best_error, best_threshold, best_codes = error, threshold, codes
    dequant = (best_codes * row_scale).contiguous()
    stored = (best_codes + 1.0).to(torch.uint8)  # {-1,0,+1} -> {0,1,2} (2 bits)
    padded, _ = _pad_columns(stored, 4)
    packed = (padded[:, 0::4] | (padded[:, 1::4] << 2) | (padded[:, 2::4] << 4) | (padded[:, 3::4] << 6)).contiguous()
    scales = row_scale.reshape(-1).float().contiguous()
    codes_bytes = int(packed.numel() * packed.element_size())
    scales_bytes = int(scales.numel() * scales.element_size())
    meta = {
        "codec": "ternary_rowscale",
        "levels": "{-1,0,+1}",
        "scales_dtype": "float32",
        "threshold": float(best_threshold),
        "codes_bytes": codes_bytes,
        "scales_bytes": scales_bytes,
    }
    return dequant, codes_bytes + scales_bytes, meta, float(best_threshold), _tensor_payload(packed, scales)


def lowrank_residual(residual: Any, maximum_rank: int, *, method: str):
    """F1 comum: residual low-rank (SVD para CASCADE, PCA para AETHER/SPECTRA)."""
    rank = min(maximum_rank, min(residual.shape))
    if method == "svd":
        u, s, v = torch.svd_lowrank(residual, q=rank, niter=4)
    else:
        u, s, v = torch.pca_lowrank(residual, q=rank, center=False, niter=4)
    u = u.float().contiguous()
    s = s.float().contiguous()
    v = v.float().contiguous()
    packed_bytes = int(sum(t.numel() * t.element_size() for t in (u, s, v)))
    meta = {
        "kind": "lowrank_svd" if method == "svd" else "lowrank_pca",
        "rank": int(s.numel()),
        "factors_dtype": "float32",
        "factors_bytes": packed_bytes,
    }
    return u, s, v, packed_bytes, meta, _tensor_payload(u, s, v)


def fit_for_architecture(arch: str, weight: Any, maximum_rank: int) -> dict[str, Any]:
    """Ajusta F0+F1 conforme a arquitetura vencedora (docs/C3_CONTRACTS_V1.md §1/§6).

    CASCADE: INT4 groupwise (grupo 32, [-8,7], escalas FP16) + residual SVD low-rank.
    RIFT: base 2-bit groupwise + refinamento 2-bit do residual (progressivo, empacotado).
    GEYSER: ZDC INT2g64 assimétrico (grupo 64, níveis {0..3}, escala (max-min)/3 FP16
    + mínimo FP16, 4 códigos/byte) + residual SVD low-rank comum.
    AETHER/SPECTRA: ternário + PCA/low-rank (caminho existente).
    Gate v0 (L2-percentile) é comum e aplicado fora desta função.
    Todos os *_packed_bytes são tamanhos reais de buffers empacotados.
    """
    arch = str(arch or "CASCADE").upper()
    threshold = None
    if arch == "CASCADE":
        f0_dequant, f0_bytes, f0_meta, f0_payload = quantize_int4_groupwise(weight)
        u, s, v, f1_bytes, f1_meta, f1_payload = lowrank_residual(weight - f0_dequant, maximum_rank, method="svd")
        f1_dequant = ((u * s) @ v.T).contiguous()
        f1_factors = (u, s, v)
        residual_rank = int(s.numel())
    elif arch == "GEYSER":
        f0_dequant, f0_bytes, f0_meta, f0_payload = quantize_zdc_int2_asymmetric_groupwise(weight, group_size=64)
        u, s, v, f1_bytes, f1_meta, f1_payload = lowrank_residual(weight - f0_dequant, maximum_rank, method="svd")
        f1_dequant = ((u * s) @ v.T).contiguous()
        f1_factors = (u, s, v)
        residual_rank = int(s.numel())
    elif arch == "RIFT":
        f0_dequant, f0_bytes, f0_meta, f0_payload = quantize_int2_groupwise(weight, role="base")
        f1_dequant, f1_bytes, f1_meta, f1_payload = quantize_int2_groupwise(weight - f0_dequant, role="refinement")
        f1_factors = None
        residual_rank = 0
    else:  # AETHER e SPECTRA: ternário + PCA/low-rank (código atual)
        f0_dequant, f0_bytes, f0_meta, threshold, f0_payload = quantize_ternary_rowscale(weight)
        u, s, v, f1_bytes, f1_meta, f1_payload = lowrank_residual(weight - f0_dequant, maximum_rank, method="pca")
        f1_dequant = ((u * s) @ v.T).contiguous()
        f1_factors = (u, s, v)
        residual_rank = int(s.numel())
    return {
        "architecture": arch,
        "f0_dequant": f0_dequant,
        "f0_packed_bytes": int(f0_bytes),
        "f0_meta": f0_meta,
        "f0_payload": f0_payload,
        "f1_dequant": f1_dequant,
        "f1_packed_bytes": int(f1_bytes),
        "f1_meta": f1_meta,
        "f1_payload": f1_payload,
        "f1_factors": f1_factors,
        "threshold": threshold,
        "residual_rank": residual_rank,
    }


# ---------------------------------------------------------------------------
# P1_WINNER_E2E_TOKS — tok/s de topo REAL baseline E candidato
# (docs/C3_CONTRACTS_V1.md §12; codec = o da ARQUITETURA SELECIONADA dinamicamente,
# via fit_for_architecture — incl. GEYSER ZDC INT2g64 assimétrico. Crib de
# C3InlineLinearModule/CascadeLinearModule de c3_methodology_auto_batteries.py:
# o W denso original fica FORA do caminho quente.)
# ---------------------------------------------------------------------------


def e2e_params_limit() -> float:
    """Limite de parâmetros do e2e (mesmo guard do G3_GEYSER_BURST); override por env."""
    raw = os.environ.get("RIFT_E2E_MAX_PARAMS", "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            print(f"[E2E] AVISO: RIFT_E2E_MAX_PARAMS inválido ({raw}); usando {E2E_MAX_PARAMS}")
    return float(E2E_MAX_PARAMS)


def find_transformer_blocks(model: Any) -> list[tuple[str, Any]]:
    """Localiza a lista de blocos transformer (crib de cascade/compiler/block_decompose.py).

    Cobre Qwen, Llama, Phi, Gemma, GPT-NeoX e fallback genérico por ModuleList.
    """
    candidate_attrs = (
        "model.layers",
        "model.model.layers",
        "transformer.h",
        "transformer.layers",
        "model.decoder.layers",
        "gpt_neox.layers",
        "language_model.model.layers",
        "model.language_model.layers",
    )
    for attr in candidate_attrs:
        node = model
        ok = True
        for part in attr.split("."):
            if not hasattr(node, part):
                ok = False
                break
            node = getattr(node, part)
        if ok and isinstance(node, (torch.nn.ModuleList, list)) and len(node) > 0:
            return [(f"{attr}.{i}", node[i]) for i in range(len(node))]
    for name, node in model.named_modules():
        if not isinstance(node, torch.nn.ModuleList) or len(node) == 0:
            continue
        if name.split(".")[-1] not in ("layers", "h", "blocks", "layer"):
            continue
        if any(isinstance(m, torch.nn.Linear) for m in node[0].modules()):
            return [(f"{name}.{i}", node[i]) for i in range(len(node))]
    return []


def set_module_by_path(root: Any, dotted: str, new_module: Any) -> None:
    """Substitui submódulo por caminho pontilhado (suporta índices numéricos)."""
    parts = dotted.split(".") if dotted else []
    if not parts:
        raise ValueError("caminho de módulo vazio")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    leaf = parts[-1]
    if leaf.isdigit():
        parent[int(leaf)] = new_module
    else:
        setattr(parent, leaf, new_module)


def collect_block_linears(block: Any, block_name: str) -> dict[str, Any]:
    """Mapa nome-completo -> nn.Linear do bloco (originais guardados ANTES da troca)."""
    out: dict[str, Any] = {}
    for name, module in block.named_modules():
        if isinstance(module, torch.nn.Linear):
            out[f"{block_name}.{name}" if name else block_name] = module
    return out


def restore_block_linears(block: Any, originals: dict[str, Any], block_name: str) -> None:
    """Devolve as nn.Linear originais ao bloco (unpatch transacional)."""
    for full, linear in originals.items():
        short = full[len(block_name) + 1:] if full.startswith(block_name + ".") else full
        try:
            set_module_by_path(block, short, linear)
        except Exception as exc:
            print(f"[E2E] AVISO ao restaurar {full}: {exc}")


def forward_logits(model: Any, inputs: dict[str, Any]) -> Any:
    """1 forward para logits (qualidade e2e); None quando o modelo não expõe logits."""
    try:
        with torch.inference_mode():
            out = model(**inputs)
        logits = getattr(out, "logits", None)
        if logits is None and isinstance(out, tuple) and out and torch.is_tensor(out[0]):
            logits = out[0]
        return logits.detach().float().cpu() if torch.is_tensor(logits) else None
    except Exception as exc:
        print(f"[E2E] AVISO forward de logits: {exc}")
        return None


def measure_generate_tok_s(model: Any, tokenizer: Any, prompt: str, device: Any, *,
                           max_new_tokens: int, warmup: int = 2, timed: int = 3) -> dict[str, Any]:
    """Tok/s REAL de model.generate (greedy) — MESMO protocolo para baseline e candidato.

    Contrato §12: >=2 warmup + >=3 medições, mediana, perf_counter_ns e
    torch.cuda.synchronize quando CUDA.
    """
    enc = tokenizer(prompt, return_tensors="pt")
    enc = {key: value.to(device) for key, value in enc.items()}
    gen_kwargs: dict[str, Any] = {"max_new_tokens": int(max_new_tokens), "do_sample": False}
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is not None:
        gen_kwargs["pad_token_id"] = pad_id
    with torch.inference_mode():
        for _ in range(max(0, int(warmup))):
            model.generate(**enc, **{**gen_kwargs, "max_new_tokens": min(8, int(max_new_tokens))})
    if device.type == "cuda":
        torch.cuda.synchronize()
    tok_s_runs: list[float] = []
    last_out = None
    n_new = 0
    for _ in range(max(1, int(timed))):
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter_ns()
        with torch.inference_mode():
            out = model.generate(**enc, **gen_kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_s = (time.perf_counter_ns() - started) / 1e9
        n_new = int(out.shape[1] - enc["input_ids"].shape[1])
        tok_s_runs.append(n_new / max(elapsed_s, 1e-9))
        last_out = out
    ordered = sorted(tok_s_runs)
    new_token_ids: list[int] = []
    if last_out is not None:
        new_token_ids = [int(t) for t in last_out[0][enc["input_ids"].shape[1]:].tolist()]
    return {
        "tok_s_median": ordered[len(ordered) // 2],
        "tok_s_runs": tok_s_runs,
        "n_new_tokens": n_new,
        "warmup_runs": int(warmup),
        "timed_runs": len(tok_s_runs),
        "greedy": True,
        "max_new_tokens": int(max_new_tokens),
        "new_token_ids": new_token_ids,
        "method": "model_generate_perf_counter_ns_median_v1",
    }


def token_exact_match(a: list[int], b: list[int]) -> dict[str, Any]:
    """Exact-match posicional dos token ids gerados (baseline vs candidato)."""
    n = min(len(a), len(b))
    if n == 0:
        return {"exact_match_rate": 0.0, "n_compared": 0, "len_baseline": len(a), "len_candidate": len(b)}
    same = sum(1 for i in range(n) if a[i] == b[i])
    return {
        "exact_match_rate": same / n,
        "n_compared": n,
        "len_baseline": len(a),
        "len_candidate": len(b),
        "length_equal": len(a) == len(b),
    }


def write_e2e_artifacts(artifacts_dir: Path, prefix: str, *, f0_payload: bytes, f1_payload: bytes) -> dict[str, Any]:
    """Grava payloads F0/F1 REAIS em <out>/artifacts/e2e e retorna bytes via os.stat."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    f0_path = artifacts_dir / f"{prefix}_f0.bin"
    f1_path = artifacts_dir / f"{prefix}_f1.bin"
    f0_path.write_bytes(f0_payload)
    f1_path.write_bytes(f1_payload)
    f0_bytes = int(os.stat(f0_path).st_size)
    f1_bytes = int(os.stat(f1_path).st_size)
    return {
        "f0_path": str(f0_path),
        "f1_path": str(f1_path),
        "f0_bytes": f0_bytes,
        "f1_bytes": f1_bytes,
        "total_bytes": f0_bytes + f1_bytes,
        "method": "binary_os_stat_v1",
    }


def run_phase_with_ram(fn):
    """Executa fn() amostrando VmRSS (~1ms) da fase; fallback getrusage só em metrics."""
    if _read_vmrss_bytes() is not None:
        sampler = VmRssSampler()
        sampler.start()
        try:
            result = fn()
        finally:
            ram = sampler.stop()
        return result, ram
    result = fn()
    peak = _getrusage_peak_bytes()
    if peak is not None:
        return result, {"method": "getrusage_peak_fallback", "max_bytes": peak, "mean_bytes": None, "samples": None}
    return result, {"method": None, "max_bytes": None, "mean_bytes": None, "samples": None}


_WINNER_INLINE_LINEAR_CLS = None


def winner_inline_linear_class():
    """Classe criada sob demanda (torch é dependência tardia neste script)."""
    global _WINNER_INLINE_LINEAR_CLS
    if _WINNER_INLINE_LINEAR_CLS is not None:
        return _WINNER_INLINE_LINEAR_CLS

    class WinnerInlineLinear(torch.nn.Module):
        """Runtime de referência Python do codec da arquitetura selecionada (F0 + Gate·F1).

        Quantiza W UMA vez no __init__ (fit_for_architecture, incl. GEYSER ZDC INT2g64
        assimétrico) e cacheia o F0 dequantizado em fp32; o W denso original fica FORA
        do caminho quente (crib de C3InlineLinearModule/CascadeLinearModule de
        c3_methodology_auto_batteries.py). Bias, quando existe, permanece fp32.
        Gate v0 comum: L2 percentil da ativação por batch (mesmo gate do P1_WINNER_F0_PLUS_LS).
        """

        def __init__(self, linear: Any, arch: str, maximum_rank: int, gate_percentile: float):
            super().__init__()
            weight = linear.weight.detach().to(dtype=torch.float32)
            fit = fit_for_architecture(arch, weight, maximum_rank)
            self.architecture = str(fit["architecture"])
            self.f0_codec = str(fit["f0_meta"]["codec"])
            self.f1_kind = str(fit["f1_meta"]["kind"])
            self.f0_packed_bytes = int(fit["f0_packed_bytes"])
            self.f1_packed_bytes = int(fit["f1_packed_bytes"])
            self.residual_rank = int(fit["residual_rank"])
            self.register_buffer("w0", fit["f0_dequant"].contiguous())
            if fit["f1_factors"] is not None:
                u, s, v = fit["f1_factors"]
                self.register_buffer("u", u.contiguous())
                self.register_buffer("s", s.contiguous())
                self.register_buffer("v", v.contiguous())
                self.register_buffer("w1", None)
            else:
                # RIFT: refinamento denso 2-bit dequantizado (progressivo)
                self.register_buffer("u", None)
                self.register_buffer("s", None)
                self.register_buffer("v", None)
                self.register_buffer("w1", fit["f1_dequant"].contiguous())
            if linear.bias is not None:
                self.register_buffer("bias_fp32", linear.bias.detach().to(dtype=torch.float32).contiguous())
            else:
                self.register_buffer("bias_fp32", None)
            self.out_features = int(weight.shape[0])
            self.in_features = int(weight.shape[1])
            self.gate_percentile = min(99.0, max(0.0, float(gate_percentile)))
            self.rows_processed = 0
            self.rows_refined = 0
            # Payloads binários reais (gravados uma vez em <out>/artifacts/e2e e descartados)
            self.f0_payload: bytes | None = fit["f0_payload"]
            self.f1_payload: bytes | None = fit["f1_payload"]

        def _residual(self, x2: Any) -> Any:
            if self.w1 is not None:
                return F.linear(x2, self.w1)
            return ((x2 @ self.v) * self.s) @ self.u.T

        def forward(self, x: Any) -> Any:
            orig_shape = x.shape
            x2 = x.reshape(-1, orig_shape[-1]).to(dtype=torch.float32)
            y = F.linear(x2, self.w0)
            gate_features = torch.linalg.vector_norm(x2, dim=1) / (float(x2.shape[1]) ** 0.5 + 1e-12)
            threshold = torch.quantile(gate_features, self.gate_percentile / 100.0)
            mask = gate_features >= threshold
            fired = int(mask.to(torch.int64).sum().item())
            if fired == int(x2.shape[0]):
                y = y + self._residual(x2)
            elif fired:
                y[mask] += self._residual(x2[mask])
            if self.bias_fp32 is not None:
                y = y + self.bias_fp32
            self.rows_processed += int(x2.shape[0])
            self.rows_refined += fired
            if y.dtype != x.dtype:
                y = y.to(dtype=x.dtype)
            return y.reshape(*orig_shape[:-1], self.out_features)

    _WINNER_INLINE_LINEAR_CLS = WinnerInlineLinear
    return _WINNER_INLINE_LINEAR_CLS


def run_e2e_tok_s_battery(recorder: "BatteryRecorder", *, model: Any, tokenizer: Any,
                          device: Any, out_dir: Path, arch: str, selection: dict[str, Any],
                          maximum_rank: int, gate_percentile: float) -> dict[str, Any] | None:
    """P1_WINNER_E2E_TOKS (contrato §12): baseline E candidato REAIS via model.generate.

    Candidato = TODAS as nn.Linear dos blocos no runtime de referência do codec da
    ARQUITETURA SELECIONADA dinamicamente. Transacional: os módulos originais são
    guardados ANTES da troca e restaurados no finally; falha em qualquer fase →
    registro FAIL e a run segue.
    """
    battery_id = "P1_WINNER_E2E_TOKS"
    n_params = sum(int(p.numel()) for p in model.parameters())
    limit = e2e_params_limit()
    if n_params > limit:
        recorder.record(
            battery_id=battery_id,
            status="SKIPPED",
            measurement_scope=(
                "e2e tok/s não executado: modelo acima do limite de parâmetros para o "
                "runtime de referência Python — velocidade não representa kernel nativo."
            ),
            quality={"full_local_gate_pass": None},
            metrics={
                "e2e": {
                    "measured": False,
                    "skipped": True,
                    "n_params": n_params,
                    "limit": int(limit),
                    "override_env": "RIFT_E2E_MAX_PARAMS",
                },
                "winner": {"architecture_executed": arch},
            },
            notes=(
                f"Modelo com {n_params / 1e9:.2f}B parâmetros excede o limite de "
                f"{limit / 1e9:.0f}e9 (mesmo guard do G3_GEYSER_BURST); o patch fp32 de "
                "referência de todas as Linears causaria OOM neste ambiente."
            ),
        )
        print(f"[E2E] {battery_id}: SKIPPED (n_params={n_params / 1e9:.2f}B > {limit / 1e9:.0f}B)")
        return None

    # Guardas pré-voo espelhadas de RIFT/AETHER: limitação de ambiente vira
    # SKIPPED (nunca FAIL) — a fila serial segue.
    def _skip_env(reason: str, extra: dict[str, Any] | None = None) -> None:
        recorder.record(
            battery_id=battery_id,
            status="SKIPPED",
            measurement_scope=(
                "e2e tok/s não executado: " + reason + " — runtime de referência "
                "Python; velocidade não representa kernel nativo."
            ),
            quality={"full_local_gate_pass": None},
            metrics={
                "e2e": {"measured": False, "skipped": True, "reason": reason, **(extra or {})},
                "winner": {"architecture_executed": arch},
            },
            notes=f"SKIPPED: {reason}. tok/s de topo permanecem null.",
        )
        print(f"[E2E] {battery_id}: SKIPPED ({reason})")

    supports_generate = callable(getattr(model, "generate", None))
    try:
        can_generate = getattr(model, "can_generate", None)
        if callable(can_generate):
            supports_generate = supports_generate and bool(can_generate())
    except Exception:
        pass
    if not supports_generate:
        _skip_env("modelo não expõe model.generate — bateria E2E não se aplica",
                  {"n_params": n_params})
        return None

    guard_params = 0
    for guard_block_name, guard_block in find_transformer_blocks(model):
        for guard_linear in collect_block_linears(guard_block, guard_block_name).values():
            guard_params += int(guard_linear.weight.numel())
    w0_cache_bytes = guard_params * 4       # F0 dequantizado fp32 cacheado
    packed_bytes_est = guard_params // 4    # payloads F0/F1 low-bit nos artefatos
    if getattr(device, "type", "") == "cuda":
        try:
            free_vram = int(torch.cuda.mem_get_info()[0])
        except Exception:
            free_vram = None
        if free_vram is not None and w0_cache_bytes > 0.8 * free_vram:
            _skip_env("VRAM insuficiente para o cache fp32 do F0 dequantizado",
                      {"w0_cache_bytes": w0_cache_bytes, "free_vram_bytes": free_vram,
                       "n_params": n_params})
            return None
    else:
        mem_available = _read_meminfo_available_bytes()
        if mem_available is not None and (w0_cache_bytes + packed_bytes_est) > 0.8 * mem_available:
            _skip_env("RAM insuficiente para o cache fp32 do F0 dequantizado (CPU)",
                      {"w0_cache_bytes": w0_cache_bytes, "packed_bytes_est": packed_bytes_est,
                       "mem_available_bytes": mem_available, "n_params": n_params})
            return None

    patched: list[tuple[Any, str, dict[str, Any]]] = []
    replaced_modules: list[Any] = []
    artifacts_dir = out_dir / "artifacts" / "e2e"
    try:
        blocks = find_transformer_blocks(model)
        if not blocks:
            raise RuntimeError("find_transformer_blocks não localizou blocos transformer")
        print(f"[E2E] baseline model.generate (greedy, max_new_tokens={E2E_MAX_NEW_TOKENS})...")
        enc = tokenizer(E2E_GENERATION_PROMPT, return_tensors="pt")
        enc = {key: value.to(device) for key, value in enc.items()}
        logits_base = forward_logits(model, enc)
        base_gen, base_ram_phase = run_phase_with_ram(
            lambda: measure_generate_tok_s(
                model, tokenizer, E2E_GENERATION_PROMPT, device,
                max_new_tokens=E2E_MAX_NEW_TOKENS, warmup=2, timed=3,
            )
        )
        print(f"[E2E] baseline {base_gen['tok_s_median']:.2f} tok/s ({base_gen['n_new_tokens']} tokens novos)")

        inline_cls = winner_inline_linear_class()
        artifact_bytes = 0
        baseline_disk = 0
        n_linears = 0

        def _patch_all() -> None:
            nonlocal artifact_bytes, baseline_disk, n_linears
            for block_name, block in blocks:
                originals = collect_block_linears(block, block_name)
                # Transacional: originais guardados ANTES de qualquer troca deste bloco.
                patched.append((block, block_name, originals))
                for full, linear in originals.items():
                    module = inline_cls(linear, arch, maximum_rank, gate_percentile)
                    art = write_e2e_artifacts(
                        artifacts_dir, full.replace(".", "_"),
                        f0_payload=module.f0_payload, f1_payload=module.f1_payload,
                    )
                    module.f0_payload = None
                    module.f1_payload = None
                    artifact_bytes += int(art["total_bytes"])
                    # Referência FP32 (numel*4): mesma base do baseline_disk_bytes
                    # dos E2E de RIFT/AETHER — os quatro *_E2E_TOKS dividem o mesmo
                    # card "Antes" no dashboard (§13.1).
                    baseline_disk += int(linear.weight.numel() * 4)
                    short = full[len(block_name) + 1:] if full.startswith(block_name + ".") else full
                    set_module_by_path(block, short, module)
                    replaced_modules.append(module)
                    n_linears += 1

        print(f"[E2E] patch de todas as Linears de {len(blocks)} blocos (codec {arch})...")
        _, patch_ram_phase = run_phase_with_ram(_patch_all)
        logits_cand = forward_logits(model, enc)
        cand_gen, cand_ram_phase = run_phase_with_ram(
            lambda: measure_generate_tok_s(
                model, tokenizer, E2E_GENERATION_PROMPT, device,
                max_new_tokens=E2E_MAX_NEW_TOKENS, warmup=2, timed=3,
            )
        )
        print(f"[E2E] candidato {cand_gen['tok_s_median']:.2f} tok/s")

        em = token_exact_match(base_gen["new_token_ids"], cand_gen["new_token_ids"])
        logits_q = (
            compute_metrics(logits_base, logits_cand)
            if (logits_base is not None and logits_cand is not None) else None
        )
        logits_cos = logits_q["cosine"] if logits_q else None
        baseline_tok_s = float(base_gen["tok_s_median"])
        candidate_tok_s = float(cand_gen["tok_s_median"])
        e2e_pass = (
            baseline_tok_s > 0 and candidate_tok_s > 0
            and logits_cos is not None and logits_cos >= 0.95
        )
        rows_processed = sum(m.rows_processed for m in replaced_modules)
        rows_refined = sum(m.rows_refined for m in replaced_modules)
        f0_codec = replaced_modules[0].f0_codec if replaced_modules else None
        f1_kind = replaced_modules[0].f1_kind if replaced_modules else None
        memory = {
            "method": (cand_ram_phase or {}).get("method") or (base_ram_phase or {}).get("method"),
            "baseline_phase": base_ram_phase,
            "candidate_phase": cand_ram_phase,
            "patch_phase": patch_ram_phase,
            "baseline_phase_scope": "model.generate baseline (modelo original completo)",
            "candidate_phase_scope": "model.generate candidato (todas as Linears dos blocos patchadas)",
        }
        recorder.record(
            battery_id=battery_id,
            status="PASS" if e2e_pass else "FAIL",
            baseline_tok_s=baseline_tok_s,
            candidate_tok_s=candidate_tok_s,
            baseline_ram_bytes=measured_phase_max(base_ram_phase),
            candidate_ram_bytes=measured_phase_max(cand_ram_phase),
            baseline_disk_bytes=baseline_disk,
            candidate_disk_bytes=artifact_bytes,
            measurement_scope=(
                "Modelo completo: baseline E candidato via model.generate (prompt PT-BR fixo, "
                f"greedy, max_new_tokens={E2E_MAX_NEW_TOKENS}, 2 warmup + 3 medições, mediana, "
                f"cuda sync); candidato = TODAS as {n_linears} nn.Linear dos {len(blocks)} blocos "
                "no runtime de referência Python — velocidade não representa kernel nativo — do "
                f"codec da arquitetura selecionada ({arch}: F0={f0_codec} + Gate·F1={f1_kind}); "
                "RAM topo = pico VmRSS por fase; disco candidato = os.stat de artefatos F0/F1 "
                "reais em artifacts/e2e."
            ),
            quality={
                "full_local_gate_pass": e2e_pass,
                "output": logits_q,
                "token_exact_match": em,
            },
            metrics={
                "e2e": {
                    "measured": True,
                    "baseline": {k: v for k, v in base_gen.items() if k != "new_token_ids"},
                    "candidate": {k: v for k, v in cand_gen.items() if k != "new_token_ids"},
                    "speedup_x": candidate_tok_s / max(baseline_tok_s, 1e-12),
                    "prompt_pt_br": E2E_GENERATION_PROMPT,
                    "max_new_tokens": E2E_MAX_NEW_TOKENS,
                    "n_params": n_params,
                    "n_blocks_patched": len(blocks),
                    "n_linears_patched": n_linears,
                    "runtime": f"python_reference_{str(arch).lower()}_codec",
                    "original_weight_on_hot_path": False,
                    "lm_head_note": "lm_head/embeddings fora dos blocos permanecem originais",
                },
                "memory": memory,
                "winner": {
                    "architecture_executed": arch,
                    "selection_basis": selection.get("selection_basis"),
                    "f0_codec": f0_codec,
                    "f1_kind": f1_kind,
                    "gate": "ACTIVATION_L2_PERCENTILE_V1",
                    "gate_percentile": min(99.0, max(0.0, float(gate_percentile))),
                    "runtime_rows_processed": rows_processed,
                    "runtime_gate_fire_rate": rows_refined / max(rows_processed, 1),
                    "native_lowbit_model_kernel": False,
                },
                "artifacts": {
                    "dir": str(artifacts_dir),
                    "total_bytes": artifact_bytes,
                    "method": "binary_os_stat_v1",
                },
            },
            notes=(
                f"E2E §12: baseline={baseline_tok_s:.2f} tok/s candidato={candidate_tok_s:.2f} "
                f"tok/s (ambos REAIS via model.generate; codec da arquitetura {arch}). "
                f"logits_cos={logits_cos if logits_cos is None else round(logits_cos, 4)} "
                f"exact_match={em['exact_match_rate']:.3f}. Módulos originais restaurados no "
                "finally (transacional)."
            ),
            comparison_role="primary",
        )
        return {
            "pass": e2e_pass,
            "baseline_tok_s": baseline_tok_s,
            "candidate_tok_s": candidate_tok_s,
            "speedup_x": candidate_tok_s / max(baseline_tok_s, 1e-12),
            "logits_cosine": logits_cos,
            "token_exact_match_rate": em["exact_match_rate"],
            "architecture_executed": arch,
        }
    except Exception as exc:
        traceback.print_exc()
        message = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, MemoryError) or "out of memory" in message.lower():
            # Paridade RIFT/AETHER: OOM em runtime é limitação de ambiente → SKIPPED.
            recorder.record(
                battery_id=battery_id,
                status="SKIPPED",
                measurement_scope=(
                    "e2e tok/s não executado: memória insuficiente durante a fase E2E; "
                    "modelo restaurado no finally."
                ),
                quality={"full_local_gate_pass": None},
                metrics={
                    "e2e": {
                        "measured": False,
                        "skipped": True,
                        "reason": "memória insuficiente durante a fase E2E",
                    },
                    "winner": {"architecture_executed": arch},
                    "error": message[:800],
                },
                notes=(
                    "SKIPPED: memória insuficiente durante quantização/patch/generate; "
                    "modelo restaurado, fila segue. " + message
                )[:1200],
            )
            print(f"[E2E] SKIPPED (memória insuficiente): {message}")
            return None
        recorder.record(
            battery_id=battery_id,
            status="FAIL",
            measurement_scope="e2e tok/s (erro em runtime); modelo restaurado no finally.",
            quality={"full_local_gate_pass": False},
            metrics={
                "e2e": {"measured": False},
                "winner": {"architecture_executed": arch},
                "error": message[:800],
            },
            notes=f"Falha na bateria e2e: {exc}"[:800],
        )
        return None
    finally:
        for block, block_name, originals in patched:
            restore_block_linears(block, originals, block_name)
        if patched:
            print("[E2E] modelo restaurado (unpatch de todos os blocos)")


def _safe_extract_tar(archive_path: Path, destination: Path) -> None:
    root = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError("Archive WINNER contém caminho fora do diretório de extração")
            if member.issym() or member.islnk() or member.isdev():
                raise RuntimeError("Archive WINNER contém link ou dispositivo não permitido")
        archive.extractall(destination, members=members, filter="data")


def download_with_limit(url: str, path: Path, maximum: int) -> None:
    request = Request(url, headers={"User-Agent": "winner-colab-benchmark/0.8"})
    with urlopen(request, timeout=90) as response, path.open("wb") as output:
        declared = int(response.headers.get("Content-Length") or 0)
        if declared > maximum:
            raise RuntimeError("Archive WINNER excede o limite permitido")
        total = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                raise RuntimeError("Archive WINNER excede o limite permitido")
            output.write(chunk)


def compile_and_test_winner(work_dir: Path, *, profile_dim: int) -> dict[str, Any]:
    if not shutil.which("cmake") or not shutil.which("g++"):
        raise RuntimeError("cmake e g++ são necessários para validar WINNER.cpp")
    archive_path = work_dir / "rift-lm.tar.gz"
    source_parent = work_dir / "source"
    source_parent.mkdir(parents=True, exist_ok=True)
    archive_url = repository_archive()
    download_with_limit(archive_url, archive_path, MAX_ARCHIVE_BYTES)
    _safe_extract_tar(archive_path, source_parent)
    # Diretório raiz do archive = "<repo>-<ref>" — o nome do repo é variável
    # (contrato §14), então o glob não pode fixar "RIFT-LM-*".
    roots = list(source_parent.glob("*/engines/winner/cpp/CMakeLists.txt"))
    if len(roots) != 1:
        raise RuntimeError("engines/winner/cpp/CMakeLists.txt não encontrado no archive publicado")
    source_dir = roots[0].parent
    build_dir = work_dir / "build"
    started = time.perf_counter()
    subprocess.run([
        "cmake", "-S", str(source_dir), "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release", "-DWINNER_NATIVE=ON", "-DBUILD_TESTING=ON",
    ], check=True)
    subprocess.run(["cmake", "--build", str(build_dir), "--parallel", "2"], check=True)
    binary = build_dir / "winner"
    self_test = subprocess.run([str(binary), "--self-test"], check=True, text=True, capture_output=True)
    profile_path = work_dir / "winner_native_profile.json"
    subprocess.run([
        str(binary), "--bench-kernels", "--dim", str(profile_dim),
        "--layers", "2", "--tokens", "4", "--output", str(profile_path),
    ], check=True)
    return {
        "source": archive_url,
        "build_seconds": time.perf_counter() - started,
        "self_test_pass": "[SELF-TEST] PASS" in self_test.stdout,
        "native_profile": json.loads(profile_path.read_text(encoding="utf-8")),
    }


class BatteryRecorder:
    def __init__(self, out_dir: Path, *, model_id: str, publish_mode: str = "off", results_endpoint: str | None = None,
                 device: str | None = None, winner_selection: dict[str, Any] | None = None):
        self.out_dir = out_dir
        self.batteries_dir = out_dir / "batteries"
        self.batteries_dir.mkdir(parents=True, exist_ok=True)
        self.model_id = model_id
        self.publish_mode = publish_mode
        self.results_endpoint = results_endpoint
        self.run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
        self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.records: list[dict[str, Any]] = []
        self.json_path = out_dir / "winner_test_batteries.json"
        self.csv_path = out_dir / "winner_test_batteries.csv"
        self.device = str(device) if device else "unknown"
        self.winner_selection = winner_selection or None
        torch_version = package_version("torch")
        # Schema v2 (docs/C3_CONTRACTS_V1.md §3)
        self.comparison_context = {
            "protocol": BENCHMARK_PROTOCOL,
            "device": self.device,
            "torch": torch_version,
            "transformers": package_version("transformers"),
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        }
        basis = f"{BENCHMARK_PROTOCOL}|{self.model_id}|{self.device}|{torch_version}"
        self.comparison_group_id = "cmp-" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:24]

    def record(self, *, battery_id: str, status: str, measurement_scope: str,
               quality: dict[str, Any], metrics: dict[str, Any], notes: str,
               comparison_role: str | None = None, baseline_ram_bytes: int | None = None,
               candidate_ram_bytes: int | None = None, baseline_disk_bytes: int | None = None,
               candidate_disk_bytes: int | None = None, baseline_tok_s: float | None = None,
               candidate_tok_s: float | None = None) -> None:
        metrics = dict(metrics or {})
        if self.winner_selection:
            # Contrato §1: todo registro do WINNER grava a seleção dinâmica
            winner_block = metrics.get("winner")
            winner_block = dict(winner_block) if isinstance(winner_block, dict) else {}
            winner_block["architecture_selected"] = self.winner_selection.get("architecture_selected")
            winner_block["selection_basis"] = self.winner_selection.get("selection_basis")
            winner_block["optimized_model_counts"] = self.winner_selection.get("optimized_model_counts")
            metrics["winner"] = winner_block
        # B0 executa o self-test C++ compilado (nativo); demais medem a referência PyTorch
        kind = "NATIVE_MEASURED" if battery_id == "B0_WINNER_CPP_BUILD_SELF_TEST" else "REFERENCE_MEASURED"
        item = {
            "schema_version": 2, "timestamp_utc": self.timestamp, "run_id": self.run_id,
            "technology": "WINNER", "model_id": self.model_id, "battery_id": battery_id,
            "benchmark_protocol": BENCHMARK_PROTOCOL,
            "comparison_group_id": self.comparison_group_id,
            "comparison_context": self.comparison_context,
            "implementation": {"kind": kind, "native": kind == "NATIVE_MEASURED", "simulated": False},
            "status": status, "measurement_scope": measurement_scope,
            "quality": quality, "metrics": metrics, "notes": notes,
            # tok/s de topo: SOMENTE a bateria e2e (§12) preenche — demais permanecem null.
            "baseline_tok_s": baseline_tok_s, "candidate_tok_s": candidate_tok_s,
            "baseline_ram_bytes": baseline_ram_bytes, "candidate_ram_bytes": candidate_ram_bytes,
            "baseline_disk_bytes": baseline_disk_bytes, "candidate_disk_bytes": candidate_disk_bytes,
            "comparison_role": comparison_role,
        }
        self.records = [record for record in self.records if record["battery_id"] != battery_id]
        self.records.append(item)
        self.records.sort(key=lambda record: record["battery_id"])
        self.json_path.write_text(json.dumps(self.records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        fields = ["timestamp_utc", "run_id", "technology", "model_id", "battery_id", "status",
                  "comparison_role", "baseline_ram_bytes", "candidate_ram_bytes",
                  "baseline_disk_bytes", "candidate_disk_bytes", "measurement_scope", "notes"]
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for record in self.records:
                writer.writerow({key: record.get(key) for key in fields})
        single = self.batteries_dir / f"{self.run_id}__{battery_id}.json"
        single.write_text(json.dumps(item, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[BATTERY] {battery_id}: gravada automaticamente -> {single}")
        # Publica imediatamente no site (não espera o fim de todas as baterias)
        try:
            publish_to_vercel(
                path=self.json_path,
                mode=self.publish_mode,
                endpoint=self.results_endpoint,
                records=list(self.records),
                quiet=False,
            )
        except ResultsPublishError as exc:
            print(f"[PUBLISH] AVISO (incremental): {exc}")


def read_setting(name: str) -> str | None:
    """Env var com fallback para Secrets do Colab (RIFT_RESULTS_ENDPOINT,
    RIFT_INGEST_TOKEN, RIFT_WINNER_ARCH, HF_TOKEN, ...)."""
    value = os.environ.get(name)
    if value and value.strip():
        return value.strip()
    try:
        from google.colab import userdata
    except Exception:
        return None
    try:
        secret = str(userdata.get(name) or "").strip()
    except Exception:
        secret = ""
    if secret:
        # Espelha no ambiente para os demais consumidores do processo
        os.environ.setdefault(name, secret)
        return secret
    return None


def publish_to_vercel(path: Path | None = None, *, mode: str, endpoint: str | None, records: list[dict[str, Any]] | None = None, quiet: bool = False) -> None:
    if mode == "off":
        if not quiet:
            print("[PUBLISH] Publicação remota desativada (--publish off).")
        return
    target = endpoint or read_setting("RIFT_RESULTS_ENDPOINT")
    token = read_setting("RIFT_INGEST_TOKEN")
    missing = []
    if not target:
        missing.append("RIFT_RESULTS_ENDPOINT")
    if not token:
        missing.append("RIFT_INGEST_TOKEN")
    if missing:
        message = "Configure " + " e ".join(missing)
        if mode == "required" and not quiet:
            raise ResultsPublishError(message)
        if not quiet:
            print(f"[PUBLISH] AVISO: {message}; resultados preservados localmente.")
        return
    if urlparse(target).scheme != "https" or len(token) < 32:
        raise ResultsPublishError("Endpoint precisa ser HTTPS e token deve ter ao menos 32 caracteres")
    if records is None:
        if path is None:
            raise ResultsPublishError("path ou records é obrigatório")
        records = json.loads(path.read_text(encoding="utf-8"))
    request = Request(
        target, data=json.dumps({"records": records}).encode("utf-8"), method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json",
                 "User-Agent": "winner-colab-publisher/0.8"},
    )
    try:
        with urlopen(request, timeout=90) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise ResultsPublishError(f"Vercel respondeu HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise ResultsPublishError(f"Falha de rede ao publicar: {exc.reason}") from exc
    publication = result.get("publication", {})
    print(f"[PUBLISH] {len(records)} registro(s) aceito(s); snapshot publicado com {publication.get('records', '?')} registro(s).")
    if publication.get("commit_url"):
        print(f"[PUBLISH] Commit: {publication['commit_url']}")


def fetch_published_history(endpoint: str | None = None) -> list[dict[str, Any]] | None:
    """Histórico publicado para a seleção dinâmica do WINNER (contrato §1).

    Tenta GET no mesmo URL de RIFT_RESULTS_ENDPOINT (resposta {records:[...]}
    ou array puro); fallback para o JSON raw do GitHub; None em falha total.
    """
    candidates: list[str] = []
    target = (endpoint or read_setting("RIFT_RESULTS_ENDPOINT") or DEFAULT_RESULTS_ENDPOINT).strip()
    if urlparse(target).scheme == "https":
        candidates.append(target)
    else:
        print(f"[winner-arch] AVISO: endpoint de histórico não-HTTPS ignorado: {target}")
    try:
        fallback_url = history_fallback_url()
    except RuntimeError as exc:
        fallback_url = None
        print(f"[winner-arch] AVISO: fallback raw do GitHub indisponível: {exc}")
    if fallback_url and fallback_url not in candidates:
        candidates.append(fallback_url)
    for url in candidates:
        try:
            request = Request(url, headers={"User-Agent": "winner-colab-benchmark/0.8", "Accept": "application/json"})
            with urlopen(request, timeout=15) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            print(f"[winner-arch] AVISO: falha ao buscar histórico em {url}: {exc}")
            continue
        records = payload.get("records") if isinstance(payload, dict) else payload
        if isinstance(records, list):
            return [record for record in records if isinstance(record, dict)]
        print(f"[winner-arch] AVISO: resposta inesperada de {url}; tentando fallback.")
    return None


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _reduction_pct(baseline: Any, candidate: Any) -> float | None:
    base = _as_float(baseline)
    cand = _as_float(candidate)
    if base is None or cand is None or base <= 0:
        return None
    return max(-10000.0, min(100.0, 100.0 * (1.0 - cand / base)))


def _normalized_score(name: str, value: Any) -> float | None:
    """Espelha normalizedMetric de api/analyze.mjs (contrato §1 — implementações espelhadas)."""
    if name == "quality_gate_pass":
        return 100.0 if value is True else (0.0 if value is False else None)
    number = _as_float(value)
    if number is None:
        return None
    if name == "output_cosine":
        return min(100.0, max(0.0, (number + 1.0) * 50.0))
    if name == "output_nrmse":
        return 100.0 * (1.0 - min(1.0, max(0.0, number / 0.1)))
    if name in ("disk_reduction_pct", "ram_reduction_pct"):
        return min(100.0, max(0.0, number))
    if name == "operation_speedup_x":
        return min(1.0, max(0.0, number)) * 100.0
    return None


def _record_score(record: dict[str, Any]) -> float | None:
    quality = record.get("quality") if isinstance(record.get("quality"), dict) else {}
    output = quality.get("output") if isinstance(quality.get("output"), dict) else {}
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    operation = metrics.get("operation") if isinstance(metrics.get("operation"), dict) else {}
    values = {
        "output_cosine": output.get("cosine"),
        "output_nrmse": output.get("nrmse"),
        "disk_reduction_pct": _reduction_pct(record.get("baseline_disk_bytes"), record.get("candidate_disk_bytes")),
        "ram_reduction_pct": _reduction_pct(record.get("baseline_ram_bytes"), record.get("candidate_ram_bytes")),
        "operation_speedup_x": operation.get("speedup_x"),
        "quality_gate_pass": quality.get("full_local_gate_pass") if isinstance(quality.get("full_local_gate_pass"), bool) else None,
    }
    weighted = 0.0
    available = 0.0
    for name, weight in SCORE_WEIGHTS.items():
        score = _normalized_score(name, values[name])
        if score is not None:
            weighted += score * weight
            available += weight
    if not available:
        return None
    coverage = available / 100.0
    return (weighted / available) * (0.65 + 0.35 * coverage)


def select_winner_architecture(records: list[dict[str, Any]]) -> tuple[str, dict[str, int]]:
    """Política determinística do WINNER dinâmico (docs/C3_CONTRACTS_V1.md §1).

    1) "modelo otimizado" = model_id distinto com registro primary, status
       PASS/EXPERIMENTAL_PASS e (quando presente) full_local_gate_pass == True;
       model_id iniciando em "synthetic/" é ignorado.
    2) vence a tecnologia com a MAIOR contagem;
    3) empate → maior score médio (SCORE_WEIGHTS, registro mais recente por
       par modelo|battery_id);
    4) empate persistente / sem dados → ordem fixa
       [CASCADE, RIFT, AETHER, SPECTRA, GEYSER].
    Registros technology="CAP" (§9) nunca entram: CAP não pertence a
    WINNER_ELIGIBLE_TECHNOLOGIES e cai no `continue` do filtro de tecnologia.
    """
    optimized: dict[str, set[str]] = {tech: set() for tech in WINNER_ELIGIBLE_TECHNOLOGIES}
    latest: dict[str, dict[tuple[str, str], dict[str, Any]]] = {tech: {} for tech in WINNER_ELIGIBLE_TECHNOLOGIES}
    for record in records or []:
        if not isinstance(record, dict):
            continue
        tech = str(record.get("technology") or "").upper()
        if tech not in optimized:
            continue
        model_id = str(record.get("model_id") or "")
        if not model_id or model_id.startswith("synthetic/"):
            continue
        if record.get("comparison_role") != "primary":
            continue
        if str(record.get("status") or "").strip().upper() not in {"PASS", "EXPERIMENTAL_PASS"}:
            continue
        quality = record.get("quality") if isinstance(record.get("quality"), dict) else {}
        if quality.get("full_local_gate_pass") is False:
            continue
        optimized[tech].add(model_id)
        key = (model_id, str(record.get("battery_id") or ""))
        previous = latest[tech].get(key)
        if previous is None or str(record.get("timestamp_utc") or "") >= str(previous.get("timestamp_utc") or ""):
            latest[tech][key] = record
    counts = {tech: len(models) for tech, models in optimized.items()}
    best_count = max(counts.values()) if counts else 0
    tied = [tech for tech in WINNER_ELIGIBLE_TECHNOLOGIES if counts.get(tech, 0) == best_count]
    if best_count == 0:
        return WINNER_PRIORITY_ORDER[0], counts
    if len(tied) > 1:
        means: dict[str, float | None] = {}
        for tech in tied:
            scores = [score for score in (_record_score(item) for item in latest[tech].values()) if score is not None]
            means[tech] = (sum(scores) / len(scores)) if scores else None
        scored = [tech for tech in tied if means[tech] is not None]
        if scored:
            top_score = max(means[tech] for tech in scored)
            top = [tech for tech in scored if abs(means[tech] - top_score) < 1e-9]
            if top:
                tied = top
    for tech in WINNER_PRIORITY_ORDER:
        if tech in tied:
            return tech, counts
    return WINNER_PRIORITY_ORDER[0], counts


def resolve_winner_selection(endpoint: str | None = None) -> dict[str, Any]:
    """Arquitetura do WINNER: override RIFT_WINNER_ARCH > histórico publicado > incumbente."""
    override = (read_setting("RIFT_WINNER_ARCH") or "").strip().upper()
    if override:
        if override in WINNER_ELIGIBLE_TECHNOLOGIES:
            print(f"[winner-arch] Override RIFT_WINNER_ARCH={override} (env_override)")
            return {"architecture_selected": override, "selection_basis": "env_override",
                    "optimized_model_counts": {}}
        print(f"[winner-arch] AVISO: RIFT_WINNER_ARCH inválido ({override}); usando seleção dinâmica.")
    records = fetch_published_history(endpoint)
    if records is None:
        print("[winner-arch] Histórico indisponível — incumbente CASCADE (default_incumbent).")
        return {"architecture_selected": "CASCADE", "selection_basis": "default_incumbent",
                "optimized_model_counts": {}}
    arch, counts = select_winner_architecture(records)
    print(f"[winner-arch] Selecionado {arch} via histórico publicado; contagens: {counts}")
    return {"architecture_selected": arch, "selection_basis": "published_history",
            "optimized_model_counts": counts}


def run_phase1(args: argparse.Namespace, native: dict[str, Any]) -> Path:
    ensure_ml_dependencies()
    model_id = normalize_huggingface_model_id(args.model)
    device = resolve_torch_device(args.device)
    selection = resolve_winner_selection(args.results_endpoint)
    arch = selection["architecture_selected"]
    print(f"[Phase1] Arquitetura WINNER selecionada: {arch} ({selection['selection_basis']})")
    print(f"[Phase1] Carregando {model_id} em {device}...")
    try:
        tokenizer, model = load_model(model_id, device=device, trust_remote_code=args.trust_remote_code)
    except Exception as load_exc:
        print(f"[Phase1] FALHA ao carregar modelo/tokenizer: {load_exc}")
        out_dir = Path(tempfile.mkdtemp(prefix="winner_phase1_fail_"))
        recorder = BatteryRecorder(
            out_dir,
            model_id=model_id,
            publish_mode=args.publish,
            results_endpoint=args.results_endpoint,
            device=str(device),
            winner_selection=selection,
        )
        recorder.record(
            battery_id="P1_LOAD_MODEL",
            status="FAIL",
            measurement_scope="model_load",
            quality={"full_local_gate_pass": False},
            metrics={"error": str(load_exc)[:800]},
            notes=(
                f"Falha ao carregar {model_id}. "
                + (
                    "Modelo gated/restrito: aceite os termos no Hugging Face e configure o Secret HF_TOKEN no Colab. "
                    if is_gated_access_error(load_exc) else
                    "Modelos diffusers/vídeo ou formato incompatível podem falhar nesta bateria CausalLM. "
                )
                + f"Detalhe: {load_exc}"
            )[:1200],
        )
        return recorder.json_path
    target_layer = resolve_linear_weight_name(model, args.target_layer)
    weight = model.state_dict()[target_layer].detach().to(device=device, dtype=torch.float32)
    try:
        x = capture_activation(model, tokenizer, target_layer.removesuffix(".weight"), device, args.prompt)
        activation_source = "real_model_activation"
    except Exception as exc:
        print(f"[Phase1] AVISO: ativação real indisponível ({exc}); usando fallback determinístico.")
        generator = torch.Generator(device=device).manual_seed(42)
        x = torch.randn((8, weight.shape[1]), generator=generator, device=device, dtype=torch.float32)
        activation_source = "synthetic_fallback"

    fit = fit_for_architecture(arch, weight, args.maximum_rank)
    f0_dequant = fit["f0_dequant"]
    f1_dequant = fit["f1_dequant"]
    f1_factors = fit["f1_factors"]
    threshold = fit["threshold"]
    with torch.inference_mode():
        y_reference = F.linear(x, weight)
        y_base = F.linear(x, f0_dequant)
        if f1_factors is not None:
            u, s, v = f1_factors
            y_residual = ((x @ v) * s) @ u.T
        else:
            y_residual = F.linear(x, f1_dequant)
        # --- Arquitetura de RAM estilo CASCADE ---
        # F0 (empacotado) sempre residente; F1 (residual low-rank ou refinamento 2-bit)
        # só entra no working set quando o gate de ativação (L2 percentil) dispara — Gate v0 comum.
        gate_features = torch.linalg.vector_norm(x, dim=1) / (float(x.shape[1]) ** 0.5 + 1e-12)
        gate_pct = float(getattr(args, "gate_percentile", 70.0))
        gate_pct = min(99.0, max(0.0, gate_pct))
        gate_threshold = torch.quantile(gate_features, gate_pct / 100.0)
        gate_mask = gate_features >= gate_threshold
        y_gated = y_base + gate_mask[:, None].to(y_residual.dtype) * y_residual
        y_full = y_base + y_residual
        reconstructed = f0_dequant + f1_dequant
        stage_rate = float(gate_mask.float().mean().item())
        relative_drift = torch.linalg.vector_norm(y_reference - y_gated, dim=1) / (
            torch.linalg.vector_norm(y_reference, dim=1) + 1e-12
        )
        drift_mean = float(relative_drift.mean().item())
        drift_max = float(relative_drift.max().item())
    print(
        f"[WINNER] Arquitetura executada: {arch} (F0={fit['f0_meta']['codec']}, F1={fit['f1_meta']['kind']}); "
        f"gate CASCADE-style: percentile={gate_pct}, "
        f"activation_rate={stage_rate:.3f}, residual_rank={fit['residual_rank']}"
    )
    q_weight_base = compute_metrics(weight, f0_dequant)
    q_weight = compute_metrics(weight, reconstructed)
    q_base = compute_metrics(y_reference, y_base)
    q_gated = compute_metrics(y_reference, y_gated)
    q_full = compute_metrics(y_reference, y_full)
    # Gate de qualidade alinhado ao CASCADE (cosine/nrmse + drift)
    quality_pass = (
        q_gated["cosine"] >= 0.98
        and q_gated["nrmse"] <= 0.10
        and drift_mean <= 0.08
    )
    if f1_factors is not None:
        u_f, s_f, v_f = f1_factors

        def _residual_term():
            return ((x @ v_f) * s_f) @ u_f.T
    else:

        def _residual_term():
            return F.linear(x, f1_dequant)

    def _gated_forward():
        yb = F.linear(x, f0_dequant)
        yr = _residual_term()
        gf = torch.linalg.vector_norm(x, dim=1) / (float(x.shape[1]) ** 0.5 + 1e-12)
        gm = gf >= gate_threshold
        return yb + gm[:, None].to(yr.dtype) * yr

    # RAM real por fase (thread VmRSS ~1ms; fallback getrusage) — contrato §3
    baseline_perf, baseline_ram_phase = benchmark_ms_with_ram(
        lambda: F.linear(x, weight), device=device, iterations=args.iterations,
    )
    base_perf, f0_ram_phase = benchmark_ms_with_ram(
        lambda: F.linear(x, f0_dequant), device=device, iterations=args.iterations,
    )
    gated_perf, gated_ram_phase = benchmark_ms_with_ram(_gated_forward, device=device, iterations=args.iterations)
    full_perf, full_ram_phase = benchmark_ms_with_ram(
        lambda: F.linear(x, f0_dequant) + _residual_term(),
        device=device, iterations=args.iterations,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    recorder = BatteryRecorder(
        out_dir, model_id=model_id, publish_mode=args.publish,
        results_endpoint=args.results_endpoint, device=str(device), winner_selection=selection,
    )
    baseline_disk = int(weight.numel() * weight.element_size())
    # Buffers empacotados REAIS (codes+scales / fatores), medidos dos tensores construídos
    base_disk = int(fit["f0_packed_bytes"])
    residual_disk = int(fit["f1_packed_bytes"])
    candidate_disk = base_disk + residual_disk  # payload total no disco
    io_bytes = int((x.numel() + y_reference.numel()) * 4)
    # Estimativas aritméticas de working set estilo CASCADE — vivem SOMENTE em
    # metrics.memory.estimated_* (contrato §3); nunca em *_ram_bytes de nível superior.
    estimated_baseline_ram = baseline_disk + io_bytes
    estimated_f0_ram = base_disk + io_bytes
    expected_residual_ram = int(round(stage_rate * residual_disk))
    estimated_candidate_ram = estimated_f0_ram + expected_residual_ram
    estimated_peak_ram = estimated_f0_ram + residual_disk  # se o gate abrir em 100% das linhas
    # RAM de nível superior: apenas VmRSS máximo medido por fase (senão null)
    baseline_ram_measured = measured_phase_max(baseline_ram_phase)
    f0_ram_measured = measured_phase_max(f0_ram_phase)
    gated_ram_measured = measured_phase_max(gated_ram_phase)
    memory_common = {
        "method": baseline_ram_phase.get("method"),
        "estimate_model": "cascade_working_set",
        "phases": {
            "baseline": baseline_ram_phase,
            "f0_only": f0_ram_phase,
            "gated": gated_ram_phase,
            "full": full_ram_phase,
        },
        "estimated_baseline_bytes": estimated_baseline_ram,
    }
    operation = {
        "metric": "linear_latency",
        "baseline_median_ms": baseline_perf["median_ms"],
        "candidate_median_ms": gated_perf["median_ms"],
        "speedup_x": baseline_perf["median_ms"] / max(gated_perf["median_ms"], 1e-12),
        "rows_processed": int(x.shape[0]),
        "device": str(device),
    }
    recorder.record(
        battery_id="B0_WINNER_CPP_BUILD_SELF_TEST", status="PASS",
        measurement_scope="Compilação nativa C++17 e --self-test; perfil nativo sintético separado da bateria do modelo.",
        quality={"full_local_gate_pass": bool(native["self_test_pass"])},
        metrics={"winner": native, "native_workload": "synthetic"},
        notes=(
            "Valida limites Q4, residual, árvore especulativa e page table; não mede o LLM end-to-end. "
            "Workload do bench C++ é sintético (runtime init_synthetic) até a execução de bundle real."
        ),
    )
    recorder.record(
        battery_id="P1_WINNER_F0_TERNARY_2BIT",
        status="EXPERIMENTAL_PASS" if q_base["cosine"] >= 0.90 else "EXPERIMENTAL_FAIL",
        baseline_ram_bytes=baseline_ram_measured, candidate_ram_bytes=f0_ram_measured,
        baseline_disk_bytes=baseline_disk, candidate_disk_bytes=base_disk,
        measurement_scope=(
            f"Single Linear op; F0 do codec da arquitetura {arch} ({fit['f0_meta']['codec']}); "
            "disk=payload F0 empacotado real; RAM topo=VmRSS máximo medido na fase F0 (null sem /proc); "
            "estimativas em metrics.memory.estimated_*."
        ),
        quality={"full_local_gate_pass": None, "weight": q_weight_base, "output": q_base},
        metrics={"operation": {**operation, "candidate_median_ms": base_perf["median_ms"],
                               "speedup_x": baseline_perf["median_ms"] / max(base_perf["median_ms"], 1e-12)},
                 "memory": {**memory_common, "estimated_candidate_bytes": estimated_f0_ram},
                 "winner": {"threshold": threshold, "residual_rank": 0, "stage_activation_rate": 0.0,
                            "architecture_executed": arch,
                            "f0_codec": fit["f0_meta"]["codec"],
                            "f0_meta": fit["f0_meta"],
                            "cpp_self_test_pass": True}},
        notes=(
            f"Arquitetura executada: {arch}. battery_id mantido por continuidade do dashboard; "
            f"o codec F0 varia por arquitetura (aqui: {fit['f0_meta']['codec']}). "
            "Working set estimado (F0 empacotado + I/O) em metrics.memory.estimated_*."
        ),
    )
    recorder.record(
        battery_id="P1_WINNER_F0_PLUS_LS", status="PASS" if quality_pass else "EXPERIMENTAL_FAIL",
        baseline_ram_bytes=baseline_ram_measured, candidate_ram_bytes=gated_ram_measured,
        baseline_disk_bytes=baseline_disk, candidate_disk_bytes=candidate_disk,
        measurement_scope=(
            f"Single Linear op; F0 {fit['f0_meta']['codec']} sempre residente + F1 {fit['f1_meta']['kind']} GATED "
            "(activation L2 percentile, Gate v0 comum); RAM topo=VmRSS máximo medido por fase "
            "(baseline vs gated; null sem /proc); Tok/s não medido."
        ),
        quality={
            "full_local_gate_pass": quality_pass,
            "weight": q_weight,
            "output": q_gated,
            "output_f0": q_base,
            "output_f0_plus_residual_always": q_full,
            "cumulative_drift_mean": drift_mean,
            "cumulative_drift_max": drift_max,
        },
        metrics={
            "operation": operation,
            "memory": {
                **memory_common,
                "estimated_candidate_bytes": estimated_candidate_ram,
                "estimated_residual_resident_bytes": expected_residual_ram,
                "estimated_peak_bytes": estimated_peak_ram,
            },
            "winner": {
                "threshold": threshold,
                "residual_rank": fit["residual_rank"],
                "stage_activation_rate": stage_rate,
                "gate": "ACTIVATION_L2_PERCENTILE_V1",
                "gate_percentile": gate_pct,
                "activation_source": activation_source,
                "architecture_executed": arch,
                "f0_codec": fit["f0_meta"]["codec"],
                "f1_kind": fit["f1_meta"]["kind"],
                "f0_meta": fit["f0_meta"],
                "f1_meta": fit["f1_meta"],
                "f0_resident_bytes": base_disk,
                "residual_bytes": residual_disk,
                "cpp_self_test_pass": bool(native["self_test_pass"]),
                "native_profile": native["native_profile"],
                "native_lowbit_model_kernel": False,
            },
        },
        notes=(
            f"Arquitetura executada: {arch} (seleção: {selection['selection_basis']}). "
            "F0 empacotado sempre residente; F1 só no working set quando o gate dispara "
            f"(rate={stage_rate:.3f}). RAM de nível superior é RSS medido por fase; "
            "estimativas aritméticas em metrics.memory.estimated_*. "
            "C++ self-test separado; kernel nativo ainda não carrega este tensor HF."
        ),
        comparison_role="primary",
    )
    # P1_WINNER_E2E_TOKS — tok/s de topo REAL baseline E candidato (contrato §12);
    # codec = o da arquitetura selecionada dinamicamente; patch transacional.
    e2e_summary = run_e2e_tok_s_battery(
        recorder,
        model=model,
        tokenizer=tokenizer,
        device=device,
        out_dir=out_dir,
        arch=arch,
        selection=selection,
        maximum_rank=args.maximum_rank,
        gate_percentile=gate_pct,
    )
    report = {
        "technology": "WINNER", "model_id": model_id, "target_layer": target_layer,
        "shape": list(weight.shape), "quality": q_gated, "performance": operation,
        "storage": {"baseline": baseline_disk, "candidate": candidate_disk}, "native": native,
        "e2e": e2e_summary,
        "winner_architecture": {
            "architecture_selected": arch,
            "selection_basis": selection["selection_basis"],
            "optimized_model_counts": selection["optimized_model_counts"],
            "f0_codec": fit["f0_meta"]["codec"],
            "f1_kind": fit["f1_meta"]["kind"],
        },
    }
    (out_dir / "winner_phase1_gain_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    del model, tokenizer, weight, fit, f0_dequant, f1_dequant, f1_factors, reconstructed
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("\n" + "=" * 78)
    print("WINNER.cpp PHASE 1 — GAIN TRACKER")
    print("=" * 78)
    print(f"Modelo                  : {model_id}")
    print(f"Arquitetura             : {arch} ({selection['selection_basis']})")
    print(f"Tensor                  : {target_layer}")
    print(f"Qualidade F0+LS         : cosine={q_gated['cosine']:.6f} / NRMSE={q_gated['nrmse']:.6f}")
    print(f"Disco                   : {baseline_disk:,} -> {candidate_disk:,} bytes")
    print(f"Linear PyTorch          : {baseline_perf['median_ms']:.4f} -> {gated_perf['median_ms']:.4f} ms")
    if e2e_summary:
        print(
            f"Tok/s e2e (REAL)        : baseline={e2e_summary['baseline_tok_s']:.2f} "
            f"candidato={e2e_summary['candidate_tok_s']:.2f} ({e2e_summary['speedup_x']:.3f}x) | "
            f"exact_match={e2e_summary['token_exact_match_rate']:.3f}"
        )
    else:
        print("Tok/s e2e               : SKIPPED/FAIL (ver registro P1_WINNER_E2E_TOKS)")
    print("C++ self-test           : PASS (latência nativa sintética separada)")
    print(f"Baterias JSON           : {recorder.json_path}")
    print("=" * 78)
    return recorder.json_path


def without_ipykernel_connection_args(argv: Iterable[str]) -> list[str]:
    values = list(argv)
    filtered = []
    index = 0
    while index < len(values):
        value = values[index]
        if value == "-f" and index + 1 < len(values):
            name = Path(values[index + 1]).name
            if name.startswith("kernel-") and name.endswith(".json"):
                index += 2
                continue
        if value.startswith("-f="):
            name = Path(value[3:]).name
            if name.startswith("kernel-") and name.endswith(".json"):
                index += 1
                continue
        filtered.append(value)
        index += 1
    return filtered


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="WINNER.cpp v0.8 native + real-tensor battery")
    parser.add_argument("--mode", choices=["self-test", "phase1"], default="phase1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--target-layer", default="auto")
    parser.add_argument("--prompt", default="Explique por que memória importa na inferência de modelos.")
    parser.add_argument("--device", default="auto", help="auto|cuda|cpu — auto usa GPU se houver, senão CPU")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--maximum-rank", type=int, default=16)
    parser.add_argument("--gate-percentile", type=float, default=70.0,
                        help="Percentil L2 da ativação para carregar residual (arquitetura CASCADE): residual só entra no working set quando o gate dispara")
    parser.add_argument("--cpp-profile-dim", type=int, default=128)
    parser.add_argument("--out", default="winner_m0_test_output")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--publish", choices=["auto", "required", "off"], default=os.environ.get("RIFT_PUBLISH_MODE", "auto"))
    parser.add_argument("--results-endpoint", default=None)
    values = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(without_ipykernel_connection_args(values))
    if args.iterations < 1 or args.maximum_rank < 1 or not 16 <= args.cpp_profile_dim <= 1024:
        parser.error("iterations/rank devem ser positivos e cpp-profile-dim deve estar entre 16 e 1024")
    try:
        with tempfile.TemporaryDirectory(prefix="winner_cpp_") as temporary:
            native = compile_and_test_winner(Path(temporary), profile_dim=args.cpp_profile_dim)
        if not native["self_test_pass"]:
            raise RuntimeError("WINNER.cpp compilou, mas o self-test não confirmou PASS")
        if args.mode == "self-test":
            print("WINNER.cpp SELF-TEST PASS")
            return
        batteries_path = run_phase1(args, native)
        publish_to_vercel(batteries_path, mode=args.publish, endpoint=args.results_endpoint)
    except ResultsPublishError as exc:
        raise SystemExit(f"[PUBLISH] ERRO: {exc}") from exc
    finally:
        cleanup_colab_workspace(label="WINNER")


if __name__ == "__main__":
    main()
