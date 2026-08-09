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
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
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

REPOSITORY_ARCHIVE_TEMPLATE = "https://github.com/programador-powershell/RIFT-LM/archive/{ref}.tar.gz"
MAX_ARCHIVE_BYTES = 25 * 1024 * 1024


class ResultsPublishError(RuntimeError):
    pass


def ensure_import(module: str, pip_name: str | None = None):
    try:
        return __import__(module)
    except ImportError:
        package = pip_name or module
        print(f"[deps] Instalando {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
        return __import__(module)


def repository_archive() -> str:
    ref = os.environ.get("RIFT_SOURCE_REF", "main").strip()
    if ref != "main" and not re.fullmatch(r"[a-fA-F0-9]{40}", ref):
        raise RuntimeError("RIFT_SOURCE_REF inválido")
    return REPOSITORY_ARCHIVE_TEMPLATE.format(ref=ref)


def ensure_ml_dependencies() -> None:
    global torch, F, AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoModelForMultimodalLM
    if torch is not None:
        return
    ensure_import("sentencepiece")
    ensure_import("tiktoken")
    # Gemma 4 Unified / modelos recentes exigem Transformers atualizado
    print("[deps] Garantindo transformers e accelerate atualizados (Gemma 4 / multimodal)...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "accelerate", "huggingface_hub"]
    )
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



def cleanup_colab_workspace(*, label: str = "battery", wipe_hf_cache: bool = False) -> None:
    """Libera artefatos temporários no Colab.

    Por padrão NÃO apaga o cache Hugging Face entre tecnologias da mesma célula
    serial (evita re-download de dezenas de GB). Wipe completo do hub só com
    wipe_hf_cache=True (final da fila / célula).
    """
    import gc
    import shutil
    import glob as _glob

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
        "/content/rift_serial_queue",
    ]
    for pattern in patterns:
        for match in _glob.glob(pattern):
            p = Path(match)
            try:
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                elif p.is_file():
                    p.unlink(missing_ok=True)
                removed.append(str(p))
            except Exception as exc:
                print(f"[cleanup] AVISO ao remover {p}: {exc}")

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


def fit_winner(weight: Any, maximum_rank: int):
    row_scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    normalized = weight / row_scale
    best_error = None
    best_threshold = None
    best_ternary = None
    for step in range(1, 19):
        threshold = step * 0.05
        ternary_codes = torch.where(
            normalized > threshold, torch.ones_like(weight),
            torch.where(normalized < -threshold, -torch.ones_like(weight), torch.zeros_like(weight)),
        )
        reconstructed = ternary_codes * row_scale
        error = float(torch.linalg.vector_norm(weight - reconstructed).item())
        if best_error is None or error < best_error:
            best_error, best_threshold, best_ternary = error, threshold, reconstructed
    residual = weight - best_ternary
    rank = min(maximum_rank, min(weight.shape))
    u, s, v = torch.pca_lowrank(residual, q=rank, center=False, niter=4)
    return best_ternary, u, s, v, float(best_threshold)


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
    roots = list(source_parent.glob("RIFT-LM-*/winner_cpp/CMakeLists.txt"))
    if len(roots) != 1:
        raise RuntimeError("winner_cpp/CMakeLists.txt não encontrado no archive publicado")
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
    def __init__(self, out_dir: Path, *, model_id: str, publish_mode: str = "off", results_endpoint: str | None = None):
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

    def record(self, *, battery_id: str, status: str, measurement_scope: str,
               quality: dict[str, Any], metrics: dict[str, Any], notes: str,
               comparison_role: str | None = None, baseline_ram_bytes: int | None = None,
               candidate_ram_bytes: int | None = None, baseline_disk_bytes: int | None = None,
               candidate_disk_bytes: int | None = None) -> None:
        item = {
            "schema_version": 1, "timestamp_utc": self.timestamp, "run_id": self.run_id,
            "technology": "WINNER", "model_id": self.model_id, "battery_id": battery_id,
            "status": status, "measurement_scope": measurement_scope,
            "quality": quality, "metrics": metrics, "notes": notes,
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
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else None


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



def run_phase1(args: argparse.Namespace, native: dict[str, Any]) -> Path:
    ensure_ml_dependencies()
    model_id = normalize_huggingface_model_id(args.model)
    device = resolve_torch_device(args.device)
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

    ternary, u, s, v, threshold = fit_winner(weight, args.maximum_rank)
    with torch.inference_mode():
        y_reference = F.linear(x, weight)
        y_base = F.linear(x, ternary)
        y_candidate = y_base + ((x @ v) * s) @ u.T
        reconstructed = ternary + (u * s) @ v.T
    q_weight_base = compute_metrics(weight, ternary)
    q_weight = compute_metrics(weight, reconstructed)
    q_base = compute_metrics(y_reference, y_base)
    q_candidate = compute_metrics(y_reference, y_candidate)
    quality_pass = q_candidate["cosine"] >= 0.98 and q_candidate["nrmse"] <= 0.10
    baseline_perf = benchmark_ms(lambda: F.linear(x, weight), device=device, iterations=args.iterations)
    candidate_perf = benchmark_ms(
        lambda: F.linear(x, ternary) + ((x @ v) * s) @ u.T,
        device=device, iterations=args.iterations,
    )
    base_perf = benchmark_ms(lambda: F.linear(x, ternary), device=device, iterations=args.iterations)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    recorder = BatteryRecorder(out_dir, model_id=model_id, publish_mode=args.publish, results_endpoint=args.results_endpoint)
    numel = int(weight.numel())
    baseline_disk = numel * 4
    base_disk = (numel + 3) // 4 + int(weight.shape[0]) * 4
    residual_disk = int((u.numel() + s.numel() + v.numel()) * 4)
    candidate_disk = base_disk + residual_disk
    io_bytes = int((x.numel() + y_reference.numel()) * 4)
    baseline_ram = baseline_disk + io_bytes
    base_ram = numel + int(weight.shape[0]) * 4 + io_bytes
    candidate_ram = base_ram + residual_disk
    operation = {
        "metric": "linear_latency", "baseline_median_ms": baseline_perf["median_ms"],
        "candidate_median_ms": candidate_perf["median_ms"],
        "speedup_x": baseline_perf["median_ms"] / candidate_perf["median_ms"],
        "rows_processed": int(x.shape[0]), "device": str(device),
    }
    recorder.record(
        battery_id="B0_WINNER_CPP_BUILD_SELF_TEST", status="PASS",
        measurement_scope="Compilação nativa C++17 e --self-test; perfil nativo sintético separado da bateria do modelo.",
        quality={"full_local_gate_pass": bool(native["self_test_pass"])}, metrics={"winner": native},
        notes="Valida limites Q4, residual, árvore especulativa e page table; não mede o LLM end-to-end.",
    )
    recorder.record(
        battery_id="P1_WINNER_F0_TERNARY_2BIT",
        status="EXPERIMENTAL_PASS" if q_base["cosine"] >= 0.90 else "EXPERIMENTAL_FAIL",
        baseline_ram_bytes=baseline_ram, candidate_ram_bytes=base_ram,
        baseline_disk_bytes=baseline_disk, candidate_disk_bytes=base_disk,
        measurement_scope="Uma operação Linear real; disco F0 em 2 bits; RAM do path de referência usa códigos int8.",
        quality={"full_local_gate_pass": None, "weight": q_weight_base, "output": q_base},
        metrics={"operation": {**operation, "candidate_median_ms": base_perf["median_ms"],
                               "speedup_x": baseline_perf["median_ms"] / base_perf["median_ms"]},
                 "winner": {"threshold": threshold, "residual_rank": 0, "cpp_self_test_pass": True}},
        notes="O armazenamento é empacotado; o kernel PyTorch desta bateria pré-decodifica F0 em FP32.",
    )
    recorder.record(
        battery_id="P1_WINNER_F0_PLUS_LS", status="PASS" if quality_pass else "EXPERIMENTAL_FAIL",
        baseline_ram_bytes=baseline_ram, candidate_ram_bytes=candidate_ram,
        baseline_disk_bytes=baseline_disk, candidate_disk_bytes=candidate_disk,
        measurement_scope="Uma operação Linear real; F0 + residual low-rank; latência PyTorch e self-test C++ reportados separadamente; Tok/s não medido.",
        quality={"full_local_gate_pass": quality_pass, "weight": q_weight, "output": q_candidate,
                 "output_base": q_base},
        metrics={"operation": operation, "winner": {
            "threshold": threshold, "residual_rank": int(s.numel()),
            "stage_activation_rate": 1.0, "activation_source": activation_source,
            "cpp_self_test_pass": bool(native["self_test_pass"]),
            "native_profile": native["native_profile"], "native_lowbit_model_kernel": False,
        }},
        notes="O C++ validado ainda não carrega diretamente este tensor do Hugging Face; nenhum ganho end-to-end ou kernel low-bit nativo é reivindicado.",
        comparison_role="primary",
    )
    report = {
        "technology": "WINNER", "model_id": model_id, "target_layer": target_layer,
        "shape": list(weight.shape), "quality": q_candidate, "performance": operation,
        "storage": {"baseline": baseline_disk, "candidate": candidate_disk}, "native": native,
    }
    (out_dir / "winner_phase1_gain_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    del model, tokenizer, weight, ternary, u, s, v, reconstructed
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("\n" + "=" * 78)
    print("WINNER.cpp PHASE 1 — GAIN TRACKER")
    print("=" * 78)
    print(f"Modelo                  : {model_id}")
    print(f"Tensor                  : {target_layer}")
    print(f"Qualidade F0+LS         : cosine={q_candidate['cosine']:.6f} / NRMSE={q_candidate['nrmse']:.6f}")
    print(f"Disco                   : {baseline_disk:,} -> {candidate_disk:,} bytes")
    print(f"Linear PyTorch          : {baseline_perf['median_ms']:.4f} -> {candidate_perf['median_ms']:.4f} ms")
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
