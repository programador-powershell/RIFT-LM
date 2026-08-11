#!/usr/bin/env python3
"""GGUF — Muse Glimmer 2-bit no Colab T4 via llama.cpp (docs/C3_CONTRACTS_V1.md §11).

Protocolo GGUF_RUNTIME_V1. Alvo padrão: unsloth/Muse-Glimmer-30B-GGUF, quant
UD-Q2_K_XL (~12–15 GB; o T4 exige llama.cpp com offload parcial CPU/GPU — o
transformers NÃO carrega os 30B em BF16 (55,7 GB) no T4).

Baterias emitidas (schema v2, publish incremental):

    B0_GGUF_RUNTIME_SETUP        (technology=GGUF)  download do binário oficial
        do llama.cpp (release PINADA por tag + verificação sha256 opcional via
        GGUF_LLAMACPP_SHA256; fallback: build com cmake) e do GGUF do quant
        escolhido via huggingface_hub (APENAS os arquivos do quant, nunca o
        repositório inteiro). Segundos reais de setup + bytes reais de disco
        (os.stat) + checagem de disco livre (--disk-budget-gb).
    P1_GGUF_E2E_TOKS             (technology=GGUF)  tok/s REAL de decode via
        llama.cpp (llama-server + HTTP /completion; fallback llama-cli), prompt
        fixo PT, >=3 medições, mediana; RAM = pico RSS da árvore de processos
        do llama.cpp amostrada a 20 ms; VRAM por PID via nvidia-smi em metrics.
        baseline_tok_s=null (BF16 de 55,7 GB não executável no T4 — sem
        comparação inventada); comparison_role=null.
    P1_GGUF_<TECH>_CODEC_TENSOR  (technology=RIFT|AETHER|CASCADE|SPECTRA)
        extrai UMA Linear REAL do GGUF (pacote pip `gguf` pinado; dequant do
        tensor para FP32) e roda o codec F0 da tecnologia + F1 low-rank (SVD)
        em NumPy puro sobre o tensor real; ativação SINTÉTICA FLAGADA
        (np.random com seed fixo) → comparison_role=null por protocolo (§3),
        mas quality.output (cosine/nrmse) é REAL sobre o tensor real e os
        packed bytes são reais.

model_id dos registros: "<model>:<quant>" (ex.:
"unsloth/Muse-Glimmer-30B-GGUF:UD-Q2_K_XL").

Dependências novas (SUJEITAS A HOMOLOGAÇÃO TI/SI, contrato §11): binário
llama.cpp (release oficial ggml-org/llama.cpp, tag pinada) e pacote pip `gguf`
(pinado gguf>=0.10,<1). pip automático SÓ no Colab (google.colab importável)
ou com RIFT_AUTO_INSTALL=1. torch/transformers NÃO são necessários aqui e são
importados apenas se já disponíveis (versões vão para comparison_context).

Publicação: POST /api/results com HTTPS obrigatório + RIFT_INGEST_TOKEN >= 32
caracteres (contrato §5). Sai com código 0 mesmo com registros FAIL; código
!= 0 apenas em crash antes de qualquer registro ser gravado.

Uso:
    python gguf_e2e_auto_batteries.py \
        --model unsloth/Muse-Glimmer-30B-GGUF --quant UD-Q2_K_XL
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

# --- dependências opcionais (nenhuma é obrigatória para todas as baterias) ---
try:
    import numpy as np
except ImportError:  # codec tensors viram SKIPPED sem numpy
    np = None  # type: ignore[assignment]
try:  # torch NÃO é necessário — só a versão para comparison_context
    import torch  # type: ignore
except ImportError:
    torch = None  # type: ignore[assignment]

BENCHMARK_PROTOCOL = "GGUF_RUNTIME_V1"
DEFAULT_ENDPOINT = "https://rift-lm.vercel.app/api/results"
DEFAULT_MODEL = "unsloth/Muse-Glimmer-30B-GGUF"
DEFAULT_QUANT = "UD-Q2_K_XL"
# Tag de release do ggml-org/llama.cpp PINADA (formato bNNNN). CONFIGURÁVEL:
# --llama-cpp-ref ou URL direta via env GGUF_LLAMACPP_URL (+ GGUF_LLAMACPP_SHA256).
DEFAULT_LLAMACPP_REF = "b6090"
LLAMACPP_REPO = "https://github.com/ggml-org/llama.cpp"
PINNED_GGUF_PIP_SPEC = "gguf>=0.10,<1"
MAX_LLAMACPP_ARCHIVE_BYTES = 3 * 1024 ** 3  # 3 GiB (builds CUDA são grandes)
RSS_SAMPLE_INTERVAL_S = 0.02  # 20 ms (contrato §11)
BASELINE_NOTE = "baseline BF16 (55.7GB) não executável no T4"

# Prompt fixo PT (mesmo texto em todas as medições do P1_GGUF_E2E_TOKS)
FIXED_PROMPT_PT = (
    "Explique, em português e em um parágrafo, por que a quantização de 2 bits "
    "reduz drasticamente a memória necessária para executar um modelo de "
    "linguagem grande, citando o papel do offload parcial entre CPU e GPU."
)

CODEC_TECHS = ("RIFT", "AETHER", "CASCADE", "SPECTRA")

# Contador global de registros gravados: exit != 0 só em crash ANTES do primeiro.
EMITTED_RECORDS = 0


# ---------------------------------------------------------------------------
# Utilidades gerais (espelham capability_eval/cascade_c2)
# ---------------------------------------------------------------------------

def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]


def _pkg_version(name: str) -> Optional[str]:
    try:
        import importlib.metadata
        return importlib.metadata.version(name)
    except Exception:
        return None


def without_ipykernel_connection_args(argv: Iterable[str]) -> List[str]:
    """Remove '-f kernel-*.json' que o ipykernel injeta no Colab (espelha M0/C3)."""
    values = list(argv)
    filtered: List[str] = []
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


def bootstrap_colab_secrets() -> None:
    """Espelha segredos do Colab (userdata) para env vars quando ausentes.

    Segredos NUNCA são gravados em arquivo — apenas ambiente do processo.
    RIFT_INGEST_TOKEN só é usado pelo publisher endurecido (HTTPS obrigatório
    + tamanho mínimo de 32 caracteres, ver publish_record).
    """
    names = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "RIFT_INGEST_TOKEN",
             "RIFT_RESULTS_ENDPOINT", "GGUF_LLAMACPP_URL", "GGUF_LLAMACPP_SHA256")
    try:
        from google.colab import userdata  # type: ignore
    except Exception:
        return
    for name in names:
        if os.environ.get(name, "").strip():
            continue
        try:
            value = str(userdata.get(name) or "").strip()
        except Exception:
            value = ""
        if value:
            os.environ[name] = value


def resolve_hf_token() -> Optional[str]:
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        v = os.environ.get(name, "").strip()
        if v:
            return v
    return None


def _colab_importable() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


def _auto_install_allowed() -> bool:
    """pip automático só no Colab ou com opt-in explícito RIFT_AUTO_INSTALL=1."""
    if os.environ.get("RIFT_AUTO_INSTALL", "").strip() == "1":
        return True
    return _colab_importable()


def ensure_gguf_module():
    """Importa o pacote pip `gguf` (PINADO; sujeito a homologação TI/SI §11).

    Instalação automática apenas no Colab ou com RIFT_AUTO_INSTALL=1.
    Retorna o módulo ou None (as baterias dependentes viram SKIPPED).
    """
    try:
        import gguf  # type: ignore
        return gguf
    except ImportError:
        pass
    if not _auto_install_allowed():
        print(f"[deps] pacote 'gguf' ausente e pip automático desativado fora do "
              f'Colab — instale manualmente: pip install "{PINNED_GGUF_PIP_SPEC}"')
        return None
    try:
        print(f'[deps] Instalando "{PINNED_GGUF_PIP_SPEC}" (pinado; homologação TI/SI §11)...')
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", PINNED_GGUF_PIP_SPEC])
        import gguf  # type: ignore
        return gguf
    except Exception as exc:
        print(f"[deps] AVISO: instalação de gguf falhou: {exc}")
        return None


def schema_v2_fields(model_id: str, device_type: str, quant: str,
                     llama_cpp_ref: str, *, native: bool) -> Dict[str, Any]:
    """Campos obrigatórios do schema v2 (docs/C3_CONTRACTS_V1.md §3)."""
    torch_v = str(getattr(torch, "__version__", "none")) if torch is not None else "none"
    raw = f"{BENCHMARK_PROTOCOL}|{model_id}|{device_type}|{torch_v}"
    kind = "NATIVE_MEASURED" if native else "REFERENCE_MEASURED"
    return {
        "schema_version": 2,
        "benchmark_protocol": BENCHMARK_PROTOCOL,
        "comparison_group_id": "cmp-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24],
        "comparison_context": {
            "protocol": BENCHMARK_PROTOCOL,
            "device": device_type,
            "torch": torch_v,
            "python": platform.python_version(),
            "runtime": "llama.cpp",
            "llama_cpp_ref": llama_cpp_ref,
            "quant": quant,
            "gguf_pkg": _pkg_version("gguf"),
        },
        "implementation": {"kind": kind, "native": native, "simulated": False},
    }


# ---------------------------------------------------------------------------
# Publisher endurecido (contrato §5): HTTPS obrigatório + token >= 32 chars
# ---------------------------------------------------------------------------

def publish_record(rec: Dict[str, Any], endpoint: Optional[str] = None) -> None:
    """Publisher endurecido: HTTPS obrigatório + token >= 32 chars (contrato §5)."""
    endpoint = endpoint or os.environ.get("RIFT_RESULTS_ENDPOINT") or DEFAULT_ENDPOINT
    token = os.environ.get("RIFT_INGEST_TOKEN") or ""
    if len(token) < 32:
        print("[publish] skip (RIFT_INGEST_TOKEN ausente ou curto <32 chars)")
        return
    if not str(endpoint).lower().startswith("https://"):
        print(f"[publish] endpoint não-HTTPS bloqueado — skip: {endpoint}")
        return
    try:
        from urllib.request import Request, urlopen
        body = json.dumps({"records": [rec]}, ensure_ascii=False).encode("utf-8")
        req = Request(endpoint, data=body, method="POST", headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "gguf-runtime-battery/1.0",
        })
        with urlopen(req, timeout=60) as resp:
            print(f"[publish] HTTP {resp.status} battery={rec.get('battery_id')}")
    except Exception as exc:
        print(f"[publish] AVISO: {exc}")


class GgufRecorder:
    """Grava JSON local (upsert por battery_id) + CSV gêmeo + publish incremental.

    Layout de artefatos idêntico ao dos irmãos (cap/cascade):
        <out>/gguf_test_batteries.json          consolidado (upsert)
        <out>/gguf_test_batteries.csv           CSV gêmeo
        <out>/batteries/<run_id>__<battery_id>.json
    """

    CSV_FIELDS = [
        "timestamp_utc", "run_id", "technology", "model_id", "battery_id", "status",
        "candidate_tok_s", "candidate_ram_bytes", "candidate_disk_bytes",
        "output_cosine", "output_nrmse", "measurement_scope",
    ]

    def __init__(self, out_dir: Path, *, model_id: str, run_id: str,
                 publish_on: bool, endpoint: Optional[str] = None):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.batteries_dir = out_dir / "batteries"
        self.batteries_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = out_dir / "gguf_test_batteries.json"
        self.csv_path = out_dir / "gguf_test_batteries.csv"
        self.model_id = model_id
        self.run_id = run_id
        self.publish_on = publish_on
        self.endpoint = endpoint
        self.records: List[Dict[str, Any]] = []
        if self.json_path.is_file():
            try:
                existing = json.loads(self.json_path.read_text(encoding="utf-8"))
                if isinstance(existing, list):
                    self.records = existing
            except Exception:
                self.records = []

    def emit(self, battery_id: str, technology: str, status: str, *,
             schema_fields: Dict[str, Any],
             baseline_tok_s: Optional[float] = None,
             candidate_tok_s: Optional[float] = None,
             baseline_ram_bytes: Optional[int] = None,
             candidate_ram_bytes: Optional[int] = None,
             baseline_disk_bytes: Optional[int] = None,
             candidate_disk_bytes: Optional[int] = None,
             quality: Optional[Dict[str, Any]] = None,
             metrics: Optional[Dict[str, Any]] = None,
             scope: str, notes: str,
             error: Optional[str] = None) -> Dict[str, Any]:
        global EMITTED_RECORDS
        metrics = dict(metrics or {})
        if error:
            metrics["error"] = str(error)[:800]
        rec = {
            "timestamp_utc": utc(),
            "run_id": self.run_id,
            "technology": technology,
            "model_id": self.model_id,
            "battery_id": battery_id,
            "status": status,
            **schema_fields,
            "comparison_role": None,  # §11: nenhuma bateria GGUF é primária
            "eligible_for_primary_ranking": False,
            "baseline_tok_s": baseline_tok_s,
            "candidate_tok_s": candidate_tok_s,
            "baseline_ram_bytes": baseline_ram_bytes,
            "candidate_ram_bytes": candidate_ram_bytes,
            "baseline_disk_bytes": baseline_disk_bytes,
            "candidate_disk_bytes": candidate_disk_bytes,
            "measurement_scope": scope,
            "quality": quality,
            "metrics": metrics,
            "notes": notes[:1200],
        }
        # upsert por battery_id no arquivo consolidado local
        self.records = [item for item in self.records if item.get("battery_id") != battery_id]
        self.records.append(rec)
        self.records.sort(key=lambda item: str(item.get("battery_id")))
        self.json_path.write_text(
            json.dumps(self.records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        single = self.batteries_dir / f"{self.run_id}__{battery_id}.json"
        single.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self._write_csv()
        EMITTED_RECORDS += 1
        print(f"[BATTERY] {battery_id}: {status} -> {single}")
        if self.publish_on:
            publish_record(rec, self.endpoint)
        return rec

    def _write_csv(self) -> None:
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.CSV_FIELDS)
            writer.writeheader()
            for item in self.records:
                output = ((item.get("quality") or {}).get("output") or {})
                writer.writerow({
                    "timestamp_utc": item.get("timestamp_utc"),
                    "run_id": item.get("run_id"),
                    "technology": item.get("technology"),
                    "model_id": item.get("model_id"),
                    "battery_id": item.get("battery_id"),
                    "status": item.get("status"),
                    "candidate_tok_s": item.get("candidate_tok_s"),
                    "candidate_ram_bytes": item.get("candidate_ram_bytes"),
                    "candidate_disk_bytes": item.get("candidate_disk_bytes"),
                    "output_cosine": output.get("cosine"),
                    "output_nrmse": output.get("nrmse"),
                    "measurement_scope": item.get("measurement_scope"),
                })


# ---------------------------------------------------------------------------
# Download + extração segura (crib de _safe_extract_tar do winner_m0)
# ---------------------------------------------------------------------------

def download_with_limit(url: str, path: Path, maximum: int) -> int:
    """Baixa url -> path com limite de bytes; retorna bytes gravados."""
    from urllib.request import Request, urlopen
    request = Request(url, headers={"User-Agent": "gguf-runtime-battery/1.0"})
    with urlopen(request, timeout=120) as response, path.open("wb") as output:
        declared = int(response.headers.get("Content-Length") or 0)
        if declared > maximum:
            raise RuntimeError(f"Archive excede o limite permitido ({declared} > {maximum})")
        total = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                raise RuntimeError(f"Archive excede o limite permitido (> {maximum} bytes)")
            output.write(chunk)
    return total


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract_tar(archive_path: Path, destination: Path) -> None:
    """Extração com guarda de path traversal (crib do winner_m0)."""
    root = destination.resolve()
    with tarfile.open(archive_path, "r:*") as archive:
        members = archive.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError("Archive llama.cpp contém caminho fora do diretório de extração")
            if member.issym() or member.islnk() or member.isdev():
                raise RuntimeError("Archive llama.cpp contém link ou dispositivo não permitido")
        archive.extractall(destination, members=members, filter="data")


def _safe_extract_zip(archive_path: Path, destination: Path) -> None:
    """Extração de zip com a mesma guarda de path traversal."""
    root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            target = (destination / info.filename).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError("Archive llama.cpp contém caminho fora do diretório de extração")
        archive.extractall(destination)


def _find_binaries(root: Path) -> Dict[str, Optional[Path]]:
    """Localiza llama-cli/llama-server sob root e marca como executáveis."""
    found: Dict[str, Optional[Path]] = {"llama-cli": None, "llama-server": None}
    for name in found:
        for candidate in sorted(root.rglob(name)):
            if candidate.is_file():
                try:
                    candidate.chmod(0o755)
                except Exception:
                    pass
                found[name] = candidate
                break
    return found


def _runtime_env(root: Path) -> Dict[str, str]:
    """Env com LD_LIBRARY_PATH apontando para as .so extraídas do release."""
    env = dict(os.environ)
    lib_dirs: List[str] = []
    for so in root.rglob("*.so*"):
        parent = str(so.parent)
        if parent not in lib_dirs:
            lib_dirs.append(parent)
    if lib_dirs:
        current = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(lib_dirs + ([current] if current else []))
    return env


def fetch_llama_cpp(ref: str, dest_dir: Path) -> Dict[str, Any]:
    """Obtém llama-cli/llama-server: release oficial pinada; fallback build cmake.

    Ordem: env GGUF_LLAMACPP_URL (com sha256 opcional em GGUF_LLAMACPP_SHA256);
    depois os nomes de asset conhecidos da release `ref` (CUDA e CPU, tar.gz e
    zip). sha256 é verificado quando fornecido; sem sha256 fica um AVISO
    registrado (verified=False).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    archives_dir = dest_dir / "archive"
    archives_dir.mkdir(parents=True, exist_ok=True)
    extract_dir = dest_dir / "bin"
    extract_dir.mkdir(parents=True, exist_ok=True)

    info: Dict[str, Any] = {
        "ref": ref, "source": None, "url": None,
        "sha256": None, "sha256_verified": False,
        "cli": None, "server": None, "env_root": str(extract_dir),
        "archive_bytes": None, "errors": [],
    }

    # binários já presentes de uma execução anterior?
    existing = _find_binaries(extract_dir)
    if existing["llama-cli"] or existing["llama-server"]:
        info.update(source="cached", cli=existing["llama-cli"], server=existing["llama-server"])
        return info

    override_url = os.environ.get("GGUF_LLAMACPP_URL", "").strip()
    expected_sha = os.environ.get("GGUF_LLAMACPP_SHA256", "").strip().lower()
    base = f"{LLAMACPP_REPO}/releases/download/{ref}"
    candidates = [override_url] if override_url else [
        f"{base}/llama-{ref}-bin-ubuntu-cuda-x64.tar.gz",
        f"{base}/llama-{ref}-bin-ubuntu-cuda-x64.zip",
        f"{base}/llama-{ref}-bin-ubuntu-x64.tar.gz",
        f"{base}/llama-{ref}-bin-ubuntu-x64.zip",
    ]

    for url in candidates:
        if not url:
            continue
        if not url.lower().startswith("https://"):
            info["errors"].append(f"URL não-HTTPS bloqueada: {url}")
            continue
        name = url.rsplit("/", 1)[-1] or "llamacpp-archive"
        archive_path = archives_dir / name
        try:
            print(f"[setup] baixando release pinada do llama.cpp: {url}")
            size = download_with_limit(url, archive_path, MAX_LLAMACPP_ARCHIVE_BYTES)
        except Exception as exc:
            info["errors"].append(f"{url}: {exc}")
            continue
        digest = sha256_file(archive_path)
        info.update(url=url, sha256=digest, archive_bytes=size)
        if expected_sha:
            if digest != expected_sha:
                info["errors"].append(
                    f"sha256 divergente: esperado {expected_sha[:16]}..., obtido {digest[:16]}...")
                archive_path.unlink(missing_ok=True)
                info.update(url=None, sha256=None, archive_bytes=None)
                continue
            info["sha256_verified"] = True
            print("[setup] sha256 verificado (GGUF_LLAMACPP_SHA256).")
        else:
            print("[setup] AVISO: GGUF_LLAMACPP_SHA256 ausente — sha256 registrado "
                  f"({digest[:16]}...) mas NÃO verificado contra valor esperado.")
        try:
            if name.endswith(".zip"):
                _safe_extract_zip(archive_path, extract_dir)
            else:
                _safe_extract_tar(archive_path, extract_dir)
        except Exception as exc:
            info["errors"].append(f"extração {name}: {exc}")
            continue
        binaries = _find_binaries(extract_dir)
        if binaries["llama-cli"] or binaries["llama-server"]:
            info.update(source="release_binary", cli=binaries["llama-cli"],
                        server=binaries["llama-server"])
            return info
        info["errors"].append(f"{name}: llama-cli/llama-server não encontrados no archive")

    # fallback: build a partir do source (mesma tag pinada)
    print("[setup] release binária indisponível — fallback: build do source com cmake...")
    try:
        built = build_llama_cpp_from_source(ref, dest_dir / "src")
        info.update(source="source_build", cli=built.get("llama-cli"),
                    server=built.get("llama-server"), env_root=str(dest_dir / "src"))
    except Exception as exc:
        info["errors"].append(f"build do source: {exc}")
    return info


def build_llama_cpp_from_source(ref: str, src_dir: Path) -> Dict[str, Optional[Path]]:
    """Clona a tag pinada e compila llama-cli/llama-server (CUDA se nvcc existir)."""
    if not shutil.which("git") or not shutil.which("cmake"):
        raise RuntimeError("git e cmake são necessários para o fallback de build")
    if not src_dir.is_dir() or not (src_dir / "CMakeLists.txt").is_file():
        if src_dir.is_dir():
            shutil.rmtree(src_dir, ignore_errors=True)
        subprocess.run([
            "git", "clone", "--depth", "1", "--branch", ref,
            f"{LLAMACPP_REPO}.git", str(src_dir),
        ], check=True)
    build_dir = src_dir / "build"
    cmake_args = ["cmake", "-S", str(src_dir), "-B", str(build_dir),
                  "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_CURL=OFF"]
    if shutil.which("nvcc"):
        cmake_args.append("-DGGML_CUDA=ON")
    subprocess.run(cmake_args, check=True)
    subprocess.run(["cmake", "--build", str(build_dir),
                    "--target", "llama-cli", "llama-server", "--parallel", "2"], check=True)
    return _find_binaries(build_dir)


def download_gguf_quant(model: str, quant: str, dest: Path,
                        token: Optional[str]) -> List[Path]:
    """Baixa APENAS os arquivos .gguf do quant escolhido (nunca o repo inteiro)."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub ausente — o launcher Colab deveria tê-lo instalado "
            "(pip install 'huggingface_hub')"
        ) from exc
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model,
        allow_patterns=[f"*{quant}*"],
        local_dir=str(dest),
        token=token,
    )
    return sorted(p for p in dest.rglob("*.gguf") if quant.lower() in p.name.lower())


def free_disk_gb(path: Path) -> float:
    usage = shutil.disk_usage(str(path))
    return usage.free / (1024 ** 3)


# ---------------------------------------------------------------------------
# Medições: RSS da árvore de processos (20 ms) + VRAM por PID via nvidia-smi
# ---------------------------------------------------------------------------

def _proc_tree_pids(root_pid: int) -> set:
    """PIDs descendentes de root_pid via /proc (Linux/Colab); {root} fora."""
    pids = {int(root_pid)}
    try:
        entries = os.listdir("/proc")
    except Exception:
        return pids
    ppid_map: Dict[int, int] = {}
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()
            after = content.rsplit(")", 1)[1].split()
            ppid_map[int(entry)] = int(after[1])
        except Exception:
            continue
    changed = True
    while changed:
        changed = False
        for pid, ppid in ppid_map.items():
            if ppid in pids and pid not in pids:
                pids.add(pid)
                changed = True
    return pids


def _pid_rss_bytes(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


class ProcessTreeRssSampler:
    """Amostra a cada 20 ms o RSS somado da árvore de processos do llama.cpp.

    peak_bytes() retorna None fora do Linux (sem /proc) — RAM de topo fica null,
    conforme regra de honestidade (§3: sem medição → null).
    """

    def __init__(self, root_pid: int, interval_s: float = RSS_SAMPLE_INTERVAL_S):
        self.root_pid = int(root_pid)
        self.interval_s = interval_s
        self._samples: List[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            pids = _proc_tree_pids(self.root_pid)
            total = sum(_pid_rss_bytes(pid) for pid in pids)
            if total > 0:
                self._samples.append(total)
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def peak_bytes(self) -> Optional[int]:
        return max(self._samples) if self._samples else None

    def info(self) -> Optional[Dict[str, Any]]:
        if not self._samples:
            return None
        return {
            "max_bytes": int(max(self._samples)),
            "mean_bytes": int(sum(self._samples) / len(self._samples)),
            "n_samples": len(self._samples),
            "method": "proc_tree_vmrss_sampling_20ms_v1",
        }


def nvidia_smi_free_vram_bytes() -> Optional[int]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        if out.returncode != 0:
            return None
        first = out.stdout.strip().splitlines()[0].strip()
        return int(float(first)) * 1024 * 1024
    except Exception:
        return None


def nvidia_smi_process_vram(pids: set) -> List[Dict[str, Any]]:
    """VRAM usada por PID (nvidia-smi --query-compute-apps), filtrada à árvore."""
    entries: List[Dict[str, Any]] = []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        if out.returncode != 0:
            return entries
        for line in out.stdout.strip().splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[0])
                mem_mb = float(parts[1])
            except ValueError:
                continue
            if pid in pids:
                entries.append({"pid": pid, "vram_bytes": int(mem_mb * 1024 * 1024)})
    except Exception:
        pass
    return entries


# ---------------------------------------------------------------------------
# -ngl auto: metadados do GGUF (block_count) + VRAM livre
# ---------------------------------------------------------------------------

def read_gguf_block_count(gguf_module, gguf_path: Path) -> Optional[int]:
    """block_count (nº de camadas) dos metadados do GGUF; None em falha."""
    try:
        reader = gguf_module.GGUFReader(str(gguf_path))
        for key, field in reader.fields.items():
            if key.endswith(".block_count"):
                try:
                    return int(field.parts[field.data[0]][0])
                except Exception:
                    value = field.contents() if hasattr(field, "contents") else None
                    if value is not None:
                        return int(value)
    except Exception as exc:
        print(f"[e2e] AVISO: leitura de block_count falhou: {exc}")
    return None


def resolve_ngl(ngl_arg: str, gguf_bytes: int,
                block_count: Optional[int]) -> Tuple[int, str]:
    """-ngl explícito ou heurística de offload parcial para a VRAM livre."""
    if str(ngl_arg).strip().lower() != "auto":
        return int(ngl_arg), "explícito (--ngl)"
    free_vram = nvidia_smi_free_vram_bytes()
    if not free_vram:
        return 0, "auto: sem GPU/nvidia-smi -> CPU (ngl=0)"
    if block_count and gguf_bytes > 0:
        per_layer = gguf_bytes / max(block_count, 1)
        layers = int((free_vram * 0.85) / max(per_layer, 1.0))
        layers = max(0, min(layers, block_count + 1))
        return layers, (f"auto: {layers} camadas (~{per_layer / 1e9:.2f} GB/camada, "
                        f"VRAM livre {free_vram / 1e9:.1f} GB, block_count={block_count})")
    if free_vram > gguf_bytes * 1.1:
        return 999, "auto: VRAM livre > tamanho do GGUF -> offload total (999)"
    return 0, "auto: sem block_count e VRAM < GGUF -> CPU (ngl=0)"


# ---------------------------------------------------------------------------
# E2E tok/s via llama-server (HTTP) com fallback llama-cli
# ---------------------------------------------------------------------------

def _http_json(url: str, payload: Optional[Dict[str, Any]] = None,
               timeout: float = 600.0) -> Any:
    from urllib.request import Request, urlopen
    if payload is None:
        req = Request(url, headers={"User-Agent": "gguf-runtime-battery/1.0"})
    else:
        req = Request(url, data=json.dumps(payload).encode("utf-8"), method="POST",
                      headers={"Content-Type": "application/json",
                               "User-Agent": "gguf-runtime-battery/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _greedy_completion(base_url: str, prompt: str, n_predict: int) -> Dict[str, Any]:
    """POST /completion (nativo) com fallback /v1/completions (OpenAI-compat)."""
    try:
        resp = _http_json(f"{base_url}/completion", {
            "prompt": prompt, "n_predict": n_predict, "temperature": 0.0,
            "top_k": 1, "top_p": 1.0, "seed": 7, "stream": False,
            "cache_prompt": False,
        })
        timings = resp.get("timings") or {}
        return {
            "text": str(resp.get("content") or ""),
            "predicted_n": timings.get("predicted_n"),
            "predicted_per_second": timings.get("predicted_per_second"),
            "endpoint": "/completion",
        }
    except Exception as exc:
        print(f"[e2e] /completion falhou ({exc}); tentando /v1/completions...")
    resp = _http_json(f"{base_url}/v1/completions", {
        "prompt": prompt, "max_tokens": n_predict, "temperature": 0.0, "seed": 7,
    })
    choices = resp.get("choices") or [{}]
    usage = resp.get("usage") or {}
    return {
        "text": str(choices[0].get("text") or ""),
        "predicted_n": usage.get("completion_tokens"),
        "predicted_per_second": None,
        "endpoint": "/v1/completions",
    }


def run_e2e_with_server(server_bin: Path, env: Dict[str, str], gguf_path: Path,
                        port: int, ngl: int, prompt: str, max_new_tokens: int,
                        n_runs: int = 3) -> Dict[str, Any]:
    """Sobe llama-server uma vez, mede >=3 runs de decode via HTTP."""
    base_url = f"http://127.0.0.1:{port}"
    cmd = [str(server_bin), "-m", str(gguf_path), "--host", "127.0.0.1",
           "--port", str(port), "-ngl", str(ngl), "-c", "2048"]
    print(f"[e2e] iniciando llama-server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    sampler = ProcessTreeRssSampler(proc.pid)
    sampler.start()
    vram_entries: List[Dict[str, Any]] = []
    runs: List[Dict[str, Any]] = []
    try:
        deadline = time.perf_counter() + 900.0  # carga de 12-15 GB pode demorar
        healthy = False
        while time.perf_counter() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"llama-server terminou cedo (rc={proc.returncode})")
            try:
                _http_json(f"{base_url}/health", timeout=5.0)
                healthy = True
                break
            except Exception:
                time.sleep(2.0)
        if not healthy:
            raise RuntimeError("llama-server não ficou saudável em 900s")
        print("[e2e] servidor saudável; warmup...")
        _greedy_completion(base_url, prompt, 8)
        for index in range(n_runs):
            t0 = time.perf_counter_ns()
            result = _greedy_completion(base_url, prompt, max_new_tokens)
            wall_s = (time.perf_counter_ns() - t0) / 1e9
            n_tokens = result.get("predicted_n")
            n_tokens = int(n_tokens) if isinstance(n_tokens, (int, float)) and n_tokens else max_new_tokens
            reported = result.get("predicted_per_second")
            if isinstance(reported, (int, float)) and reported > 0:
                tok_s = float(reported)
                source = "llama.cpp timings.predicted_per_second"
            else:
                tok_s = n_tokens / max(wall_s, 1e-9)
                source = "wall-clock / tokens gerados"
            runs.append({"run": index + 1, "tok_s": tok_s, "wall_s": wall_s,
                         "n_tokens": n_tokens, "timing_source": source,
                         "endpoint": result["endpoint"]})
            print(f"[e2e] run {index + 1}/{n_runs}: {tok_s:.2f} tok/s "
                  f"({n_tokens} tokens, wall {wall_s:.2f}s, {source})")
        vram_entries = nvidia_smi_process_vram(_proc_tree_pids(proc.pid))
    finally:
        sampler.stop()
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()
    return {"mode": "llama-server", "runs": runs, "rss": sampler.info(),
            "vram_per_pid": vram_entries}


_CLI_TIMING_RE = re.compile(
    r"(prompt\s+)?eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*(?:runs|tokens)"
    r"[^\n]*?([0-9.]+)\s*tokens per second", re.IGNORECASE)


def run_e2e_with_cli(cli_bin: Path, env: Dict[str, str], gguf_path: Path,
                     ngl: int, prompt: str, max_new_tokens: int,
                     n_runs: int = 3) -> Dict[str, Any]:
    """Fallback: llama-cli por run (recarrega o modelo; usa o timing do decode).

    O tok/s por run vem do timing de decode impresso pelo llama.cpp
    (`eval time ... tokens per second`, excluindo prompt eval e a carga do
    modelo); sem timing parseável cai para tokens/wall-clock do processo
    inteiro (honesto porém pessimista — inclui a carga; anotado por run).
    """
    runs: List[Dict[str, Any]] = []
    rss_info: Optional[Dict[str, Any]] = None
    vram_entries: List[Dict[str, Any]] = []
    base_cmd = [str(cli_bin), "-m", str(gguf_path), "-p", prompt,
                "-n", str(max_new_tokens), "--temp", "0", "--top-k", "1",
                "--seed", "7", "-ngl", str(ngl)]
    for index in range(n_runs):
        for extra in (["-no-cnv", "--no-display-prompt"], []):
            cmd = base_cmd + extra
            t0 = time.perf_counter_ns()
            proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
            sampler = ProcessTreeRssSampler(proc.pid)
            sampler.start()
            vram_probe = threading.Timer(
                20.0, lambda: vram_entries.extend(
                    nvidia_smi_process_vram(_proc_tree_pids(proc.pid))))
            vram_probe.daemon = True
            vram_probe.start()
            try:
                stdout, _ = proc.communicate(timeout=3600)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, _ = proc.communicate()
            finally:
                sampler.stop()
                vram_probe.cancel()
            wall_s = (time.perf_counter_ns() - t0) / 1e9
            if proc.returncode != 0:
                if extra:
                    # flags novas podem não existir na release pinada: repete sem elas
                    continue
                # run com falha real: descartado (mediana exige medições válidas)
                print(f"[e2e] run {index + 1}/{n_runs}: llama-cli rc={proc.returncode} "
                      "— medição descartada")
                break
            info = sampler.info()
            if info and (rss_info is None or info["max_bytes"] > rss_info["max_bytes"]):
                rss_info = info
            tok_s = None
            n_tokens = max_new_tokens
            source = None
            for match in _CLI_TIMING_RE.finditer(stdout or ""):
                if match.group(1):
                    continue  # linha de prompt eval
                n_tokens = int(match.group(3))
                tok_s = float(match.group(4))
                source = "llama.cpp eval timing (decode)"
            if tok_s is None:
                tok_s = n_tokens / max(wall_s, 1e-9)
                source = "wall-clock do processo (inclui carga do modelo)"
            runs.append({"run": index + 1, "tok_s": tok_s, "wall_s": wall_s,
                         "n_tokens": n_tokens, "timing_source": source,
                         "returncode": proc.returncode})
            print(f"[e2e] run {index + 1}/{n_runs}: {tok_s:.2f} tok/s ({source})")
            break
    return {"mode": "llama-cli", "runs": runs, "rss": rss_info,
            "vram_per_pid": vram_entries}


# ---------------------------------------------------------------------------
# Codecs F0 em NumPy puro (portes fiéis do winner_m0; sem torch) + F1 SVD
# ---------------------------------------------------------------------------

def _pad_columns_np(matrix: "np.ndarray", multiple: int) -> Tuple["np.ndarray", int]:
    rows, cols = matrix.shape
    padded_cols = ((cols + multiple - 1) // multiple) * multiple
    if padded_cols == cols:
        return np.ascontiguousarray(matrix), cols
    out = np.zeros((rows, padded_cols), dtype=matrix.dtype)
    out[:, :cols] = matrix
    return out, cols


def quantize_int4_groupwise_np(weight: "np.ndarray", group_size: int = 32):
    """F0 do CASCADE: INT4 groupwise signed [-8,7], grupo 32, escalas FP16."""
    padded, cols = _pad_columns_np(weight.astype(np.float32), group_size)
    rows = padded.shape[0]
    groups = padded.reshape(rows, -1, group_size)
    scales = np.maximum(np.abs(groups).max(axis=2) / 7.0, 1e-12).astype(np.float16)
    scale_f32 = scales.astype(np.float32)[:, :, None]
    codes = np.clip(np.round(groups / scale_f32), -8, 7).astype(np.int8)
    nibbles = (codes.astype(np.int16) + 8).astype(np.uint8).reshape(rows, -1)
    packed = np.ascontiguousarray(nibbles[:, 0::2] | (nibbles[:, 1::2] << 4))
    dequant = (codes.astype(np.float32) * scale_f32).reshape(rows, -1)[:, :cols]
    meta = {"codec": "int4_groupwise_g32", "group_size": group_size,
            "levels": "signed[-8,7]", "scales_dtype": "float16",
            "codes_bytes": int(packed.nbytes), "scales_bytes": int(scales.nbytes)}
    return np.ascontiguousarray(dequant), int(packed.nbytes + scales.nbytes), meta


def quantize_int2_groupwise_np(weight: "np.ndarray", group_size: int = 32):
    """F0 do RIFT: 2 bits/peso groupwise, 4 níveis simétricos (±0.5/±1.5 × step)."""
    padded, cols = _pad_columns_np(weight.astype(np.float32), group_size)
    rows = padded.shape[0]
    groups = padded.reshape(rows, -1, group_size)
    steps = np.maximum(np.abs(groups).max(axis=2) / 1.5, 1e-12).astype(np.float16)
    step_f32 = steps.astype(np.float32)[:, :, None]
    codes = np.clip(np.round(groups / step_f32 + 1.5), 0, 3).astype(np.uint8)
    flat = codes.reshape(rows, -1)
    packed = np.ascontiguousarray(
        flat[:, 0::4] | (flat[:, 1::4] << 2) | (flat[:, 2::4] << 4) | (flat[:, 3::4] << 6))
    dequant = ((codes.astype(np.float32) - 1.5) * step_f32).reshape(rows, -1)[:, :cols]
    meta = {"codec": "int2_groupwise_g32", "group_size": group_size,
            "levels": "symmetric4(+-0.5,+-1.5)xstep", "scales_dtype": "float16",
            "codes_bytes": int(packed.nbytes), "scales_bytes": int(steps.nbytes)}
    return np.ascontiguousarray(dequant), int(packed.nbytes + steps.nbytes), meta


def quantize_ternary_rowscale_np(weight: "np.ndarray"):
    """F0 do AETHER/SPECTRA: ternário {-1,0,+1}, escala por linha, busca de limiar."""
    weight = weight.astype(np.float32)
    row_scale = np.maximum(np.abs(weight).max(axis=1, keepdims=True), 1e-12)
    normalized = weight / row_scale
    best_error = None
    best_threshold = None
    best_codes = None
    for step in range(1, 19):
        threshold = step * 0.05
        codes = np.where(normalized > threshold, 1.0,
                         np.where(normalized < -threshold, -1.0, 0.0)).astype(np.float32)
        error = float(np.linalg.norm(weight - codes * row_scale))
        if best_error is None or error < best_error:
            best_error, best_threshold, best_codes = error, threshold, codes
    dequant = np.ascontiguousarray(best_codes * row_scale)
    stored = (best_codes + 1.0).astype(np.uint8)  # {-1,0,+1} -> {0,1,2} (2 bits)
    padded, _ = _pad_columns_np(stored, 4)
    packed = np.ascontiguousarray(
        padded[:, 0::4] | (padded[:, 1::4] << 2) | (padded[:, 2::4] << 4) | (padded[:, 3::4] << 6))
    scales = row_scale.reshape(-1).astype(np.float32)
    meta = {"codec": "ternary_rowscale", "levels": "{-1,0,+1}",
            "scales_dtype": "float32", "threshold": float(best_threshold),
            "codes_bytes": int(packed.nbytes), "scales_bytes": int(scales.nbytes)}
    return dequant, int(packed.nbytes + scales.nbytes), meta


def svd_lowrank_np(matrix: "np.ndarray", rank: int, niter: int = 2, seed: int = 0):
    """SVD low-rank randomizada em NumPy (análoga a torch.svd_lowrank)."""
    rng = np.random.default_rng(seed)
    rows, cols = matrix.shape
    q = max(1, min(rank, min(rows, cols)))
    omega = rng.standard_normal((cols, q)).astype(np.float32)
    y = matrix @ omega
    y, _ = np.linalg.qr(y)
    for _ in range(niter):
        z = matrix.T @ y
        z, _ = np.linalg.qr(z)
        y = matrix @ z
        y, _ = np.linalg.qr(y)
    b = y.T @ matrix
    u_small, s, vt = np.linalg.svd(b, full_matrices=False)
    u = y @ u_small
    return u.astype(np.float32), s.astype(np.float32), vt.astype(np.float32)


def cosine_nrmse_np(a: "np.ndarray", b: "np.ndarray") -> Dict[str, float]:
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    denom = float(np.linalg.norm(a)) + 1e-12
    cos = float(np.dot(a, b) / max(float(np.linalg.norm(a)) * float(np.linalg.norm(b)), 1e-12))
    nrmse = float(np.linalg.norm(a - b)) / denom
    return {"cosine": cos, "nrmse": nrmse}


TECH_CODECS: Dict[str, Tuple[str, Callable[..., Any]]] = {
    "RIFT": ("int2_groupwise_g32", quantize_int2_groupwise_np),
    "AETHER": ("ternary_rowscale", quantize_ternary_rowscale_np),
    "CASCADE": ("int4_groupwise_g32", quantize_int4_groupwise_np),
    "SPECTRA": ("ternary_rowscale", quantize_ternary_rowscale_np),
}

TENSOR_NAME_PREFERENCES = ("attn_output", "attn_q", "attn_k", "attn_v",
                           "ffn_down", "ffn_up", "ffn_gate")


def select_gguf_tensor(reader, max_elements: int):
    """UMA Linear 2D grande (projeção attn/ffn) do GGUF; None se nada servir."""
    best_key = None
    best_tensor = None
    for tensor in reader.tensors:
        name = str(tensor.name)
        if not name.endswith(".weight"):
            continue
        dims = [int(d) for d in tensor.shape]
        if len(dims) != 2 or min(dims) < 1024:
            continue
        elements = dims[0] * dims[1]
        if elements > max_elements:
            continue
        for pref_index, pref in enumerate(TENSOR_NAME_PREFERENCES):
            if pref in name:
                key = (pref_index, elements, name)
                if best_key is None or key < best_key:
                    best_key, best_tensor = key, tensor
                break
    return best_tensor


def dequantize_gguf_tensor(gguf_module, tensor) -> "np.ndarray":
    """Dequantiza o tensor do GGUF para FP32 2D; levanta em quant não suportado."""
    data = gguf_module.quants.dequantize(tensor.data, tensor.tensor_type)
    dims = [int(d) for d in tensor.shape]
    expected = dims[0] * dims[1]
    flat = np.asarray(data, dtype=np.float32).reshape(-1)
    if flat.size != expected:
        raise RuntimeError(
            f"dequantize retornou {flat.size} elementos (esperado {expected})")
    # GGUF guarda ne[0] como dimensão mais rápida -> numpy shape invertida
    return np.ascontiguousarray(flat.reshape(dims[1], dims[0]))


# ---------------------------------------------------------------------------
# Baterias
# ---------------------------------------------------------------------------

def run_setup_battery(args, recorder: GgufRecorder, model_id: str,
                      device_type: str, out_dir: Path,
                      hf_token: Optional[str]) -> Dict[str, Any]:
    """B0_GGUF_RUNTIME_SETUP: binário pinado + GGUF do quant + medições reais."""
    schema = schema_v2_fields(model_id, device_type, args.quant,
                              args.llama_cpp_ref, native=True)
    scope = ("setup real GGUF_RUNTIME_V1: download do binário llama.cpp "
             f"(release pinada {args.llama_cpp_ref}, sha256 quando fornecido) + "
             f"GGUF do quant {args.quant} via huggingface_hub "
             "(allow_patterns só do quant); segundos de setup e bytes de disco "
             "reais (os.stat); checagem de disco livre")
    started = time.perf_counter()
    state: Dict[str, Any] = {"ok": False, "gguf_files": [], "cli": None,
                             "server": None, "env": dict(os.environ),
                             "gguf_bytes": 0, "llama_info": None}

    free_gb = free_disk_gb(out_dir)
    if free_gb < float(args.disk_budget_gb):
        recorder.emit(
            "B0_GGUF_RUNTIME_SETUP", "GGUF", "FAIL",
            schema_fields=schema,
            metrics={"setup": {"free_disk_gb": round(free_gb, 2),
                               "disk_budget_gb": float(args.disk_budget_gb),
                               "disk_ok": False}},
            scope=scope,
            notes=(f"Disco livre insuficiente ({free_gb:.1f} GB < orçamento "
                   f"{args.disk_budget_gb} GB) — downloads não iniciados. "
                   "Reduza --disk-budget-gb apenas se souber o que está fazendo."),
            error="disco livre abaixo do orçamento")
        return state

    errors: List[str] = []
    llama_info: Dict[str, Any] = {}
    try:
        llama_info = fetch_llama_cpp(args.llama_cpp_ref, out_dir / "llamacpp")
        state["llama_info"] = llama_info
        state["cli"] = llama_info.get("cli")
        state["server"] = llama_info.get("server")
        state["env"] = _runtime_env(Path(llama_info.get("env_root") or out_dir))
        if not (state["cli"] or state["server"]):
            errors.append("llama-cli/llama-server indisponíveis (release e build falharam)")
    except Exception as exc:
        errors.append(f"llama.cpp: {exc}")

    gguf_files: List[Path] = []
    try:
        gguf_files = download_gguf_quant(args.model, args.quant, out_dir / "gguf", hf_token)
        if not gguf_files:
            errors.append(f"nenhum arquivo .gguf do quant {args.quant} encontrado em {args.model}")
    except Exception as exc:
        errors.append(f"download GGUF: {exc}")
    state["gguf_files"] = gguf_files

    gguf_bytes = 0
    for path in gguf_files:
        try:
            gguf_bytes += int(os.stat(path).st_size)
        except OSError:
            pass
    state["gguf_bytes"] = gguf_bytes
    setup_seconds = time.perf_counter() - started
    free_after_gb = free_disk_gb(out_dir)
    ok = bool(gguf_files) and bool(state["cli"] or state["server"])
    state["ok"] = ok

    binary_bytes = None
    if llama_info.get("archive_bytes"):
        binary_bytes = int(llama_info["archive_bytes"])

    recorder.emit(
        "B0_GGUF_RUNTIME_SETUP", "GGUF", "PASS" if ok else "FAIL",
        schema_fields=schema,
        candidate_disk_bytes=gguf_bytes if gguf_bytes else None,
        metrics={"setup": {
            "setup_seconds": round(setup_seconds, 3),
            "free_disk_gb_before": round(free_gb, 2),
            "free_disk_gb_after": round(free_after_gb, 2),
            "disk_budget_gb": float(args.disk_budget_gb),
            "disk_ok": True,
            "llama_cpp": {
                "ref": args.llama_cpp_ref,
                "source": llama_info.get("source"),
                "url": llama_info.get("url"),
                "sha256": llama_info.get("sha256"),
                "sha256_verified": bool(llama_info.get("sha256_verified")),
                "archive_bytes": binary_bytes,
                "cli": str(state["cli"]) if state["cli"] else None,
                "server": str(state["server"]) if state["server"] else None,
                "errors": (llama_info.get("errors") or [])[:6],
            },
            "gguf": {
                "repo": args.model,
                "quant": args.quant,
                "files": [{"name": p.name, "bytes": int(os.stat(p).st_size)}
                          for p in gguf_files],
                "total_bytes": gguf_bytes,
            },
        }},
        scope=scope,
        notes=(f"Setup {'concluído' if ok else 'INCOMPLETO'} em {setup_seconds:.1f}s; "
               f"llama.cpp={llama_info.get('source') or 'indisponível'} "
               f"(ref pinada {args.llama_cpp_ref}, sha256 "
               f"{'verificado' if llama_info.get('sha256_verified') else 'NÃO verificado — defina GGUF_LLAMACPP_SHA256'}); "
               f"GGUF {args.quant}: {len(gguf_files)} arquivo(s), "
               f"{gguf_bytes / 1e9:.2f} GB reais em disco. "
               "Dependências pinadas sujeitas a homologação TI/SI (§11)."),
        error="; ".join(errors) if errors else None)
    return state


def run_e2e_battery(args, recorder: GgufRecorder, model_id: str,
                    device_type: str, setup: Dict[str, Any],
                    gguf_module) -> None:
    """P1_GGUF_E2E_TOKS: tok/s REAL de decode via llama.cpp (mediana de >=3)."""
    schema = schema_v2_fields(model_id, device_type, args.quant,
                              args.llama_cpp_ref, native=True)
    scope = ("tok/s REAL de decode do modelo completo via llama.cpp no runtime "
             "nativo (llama-server /completion; fallback llama-cli), prompt fixo "
             "PT, geração greedy, mediana de >=3 medições; RAM=pico RSS da árvore "
             "de processos llama.cpp amostrada a 20 ms; VRAM por PID (nvidia-smi) "
             f"em metrics; {BASELINE_NOTE} -> baseline_tok_s=null, sem comparação")
    if not setup.get("ok"):
        recorder.emit(
            "P1_GGUF_E2E_TOKS", "GGUF", "SKIPPED", schema_fields=schema,
            scope=scope,
            notes=("SKIPPED: setup incompleto (B0_GGUF_RUNTIME_SETUP FAIL) — "
                   "sem binário llama.cpp e/ou sem GGUF do quant. " + BASELINE_NOTE))
        return
    gguf_files: List[Path] = setup["gguf_files"]
    main_gguf = gguf_files[0]
    gguf_bytes = int(setup["gguf_bytes"])
    try:
        block_count = read_gguf_block_count(gguf_module, main_gguf) if gguf_module else None
        ngl, ngl_basis = resolve_ngl(args.ngl, gguf_bytes, block_count)
        print(f"[e2e] -ngl {ngl} ({ngl_basis})")
        if setup.get("server"):
            result = run_e2e_with_server(
                setup["server"], setup["env"], main_gguf, int(args.server_port),
                ngl, FIXED_PROMPT_PT, int(args.max_new_tokens))
        elif setup.get("cli"):
            result = run_e2e_with_cli(
                setup["cli"], setup["env"], main_gguf, ngl,
                FIXED_PROMPT_PT, int(args.max_new_tokens))
        else:
            raise RuntimeError("nenhum binário llama.cpp disponível")
        runs = result["runs"]
        if len(runs) < 3:
            raise RuntimeError(f"apenas {len(runs)} medições concluídas (mínimo 3)")
        median_tok_s = float(statistics.median([run["tok_s"] for run in runs]))
        rss = result.get("rss")
        vram = result.get("vram_per_pid") or []
        recorder.emit(
            "P1_GGUF_E2E_TOKS", "GGUF", "PASS", schema_fields=schema,
            baseline_tok_s=None,  # honestidade: sem baseline executável no T4
            candidate_tok_s=median_tok_s,
            candidate_ram_bytes=(rss or {}).get("max_bytes"),
            candidate_disk_bytes=gguf_bytes,
            metrics={
                "e2e": {
                    "measured": True,
                    "median_tok_s": median_tok_s,
                    "runs": runs,
                    "n_runs": len(runs),
                    "mode": result["mode"],
                    "prompt": FIXED_PROMPT_PT,
                    "max_new_tokens": int(args.max_new_tokens),
                    "ngl": ngl,
                    "ngl_basis": ngl_basis,
                    "block_count": block_count,
                    "baseline_note": BASELINE_NOTE,
                },
                "memory": {"method": (rss or {}).get("method"), "rss_phase": rss,
                           "vram_per_pid": vram},
            },
            scope=scope,
            notes=(f"Decode REAL {median_tok_s:.2f} tok/s (mediana de {len(runs)} runs, "
                   f"{result['mode']}, -ngl {ngl}) para {model_id}. "
                   f"{BASELINE_NOTE}; comparison_role=null (sem comparação inventada). "
                   f"RSS pico {'%.2f GB' % ((rss or {}).get('max_bytes', 0) / 1e9) if rss else 'não medido (fora do Linux)'}."))
    except Exception as exc:
        traceback.print_exc()
        recorder.emit(
            "P1_GGUF_E2E_TOKS", "GGUF", "FAIL", schema_fields=schema,
            candidate_disk_bytes=gguf_bytes,
            scope=scope,
            notes=f"FAIL de infraestrutura na medição e2e; {BASELINE_NOTE}.",
            error=str(exc))


def run_codec_tensor_batteries(args, recorder: GgufRecorder, model_id: str,
                               device_type: str, setup: Dict[str, Any],
                               gguf_module) -> None:
    """P1_GGUF_<TECH>_CODEC_TENSOR ×4: F0+F1 NumPy sobre UMA Linear real do GGUF."""
    def _skip_all(reason: str) -> None:
        for tech in CODEC_TECHS:
            schema = schema_v2_fields(model_id, device_type, args.quant,
                                      args.llama_cpp_ref, native=False)
            recorder.emit(
                f"P1_GGUF_{tech}_CODEC_TENSOR", tech, "SKIPPED",
                schema_fields=schema,
                scope=codec_scope(tech),
                notes=f"SKIPPED: {reason}")

    if np is None:
        _skip_all("numpy indisponível — codecs F0/F1 são NumPy puro")
        return
    if gguf_module is None:
        _skip_all(f'pacote pip "gguf" indisponível (pinado {PINNED_GGUF_PIP_SPEC}; '
                  "instalação Colab-gated, homologação TI/SI §11)")
        return
    if not setup.get("gguf_files"):
        _skip_all("GGUF do quant não baixado (ver B0_GGUF_RUNTIME_SETUP)")
        return

    main_gguf: Path = setup["gguf_files"][0]
    try:
        reader = gguf_module.GGUFReader(str(main_gguf))
        tensor = select_gguf_tensor(reader, int(args.codec_max_elements))
        if tensor is None:
            _skip_all("nenhum tensor 2D attn/ffn dentro do limite "
                      f"--codec-max-elements={args.codec_max_elements}")
            return
        tensor_type = str(getattr(tensor.tensor_type, "name", tensor.tensor_type))
        weight = dequantize_gguf_tensor(gguf_module, tensor)
    except Exception as exc:
        traceback.print_exc()
        _skip_all(f"dequantização do GGUF não suportada para este quant "
                  f"({exc})")
        return

    tensor_name = str(tensor.name)
    rows, cols = int(weight.shape[0]), int(weight.shape[1])
    dense_bytes = int(weight.size * 4)
    print(f"[codec] tensor real: {tensor_name} shape=({rows},{cols}) "
          f"tipo GGUF={tensor_type} ({dense_bytes / 1e6:.1f} MB fp32)")

    # ativação SINTÉTICA FLAGADA (np.random com seed fixo) — §3: rebaixa o registro
    rng = np.random.default_rng(20260810)
    x = rng.standard_normal((32, cols)).astype(np.float32)
    y_ref = x @ weight.T

    for tech in CODEC_TECHS:
        codec_name, codec_fn = TECH_CODECS[tech]
        schema = schema_v2_fields(model_id, device_type, args.quant,
                                  args.llama_cpp_ref, native=False)
        try:
            started = time.perf_counter_ns()
            f0_dequant, f0_bytes, f0_meta = codec_fn(weight)
            residual = weight - f0_dequant
            u, s, vt = svd_lowrank_np(residual, int(args.codec_rank), niter=2, seed=0)
            f1_dequant = (u * s) @ vt
            f1_bytes = int(u.nbytes + s.nbytes + vt.nbytes)
            w_hat = f0_dequant + f1_dequant
            elapsed_s = (time.perf_counter_ns() - started) / 1e9
            y_hat = x @ w_hat.T
            output_q = cosine_nrmse_np(y_ref, y_hat)
            weight_q = cosine_nrmse_np(weight, w_hat)
            packed_bytes = int(f0_bytes + f1_bytes)
            recorder.emit(
                f"P1_GGUF_{tech}_CODEC_TENSOR", tech, "PASS",
                schema_fields=schema,
                baseline_disk_bytes=dense_bytes,
                candidate_disk_bytes=packed_bytes,
                quality={"output": output_q},
                metrics={
                    "codec": {
                        "technology": tech,
                        "f0": f0_meta,
                        "f1": {"kind": "lowrank_svd_np", "rank": int(s.size),
                               "factors_dtype": "float32", "factors_bytes": f1_bytes},
                        "f0_packed_bytes": int(f0_bytes),
                        "f1_packed_bytes": f1_bytes,
                        "total_packed_bytes": packed_bytes,
                        "dense_fp32_bytes": dense_bytes,
                        "fit_seconds": round(elapsed_s, 3),
                        "weight_cosine": weight_q["cosine"],
                        "weight_nrmse": weight_q["nrmse"],
                    },
                    "tensor": {
                        "name": tensor_name,
                        "gguf_type": tensor_type,
                        "shape": [rows, cols],
                        "source_file": main_gguf.name,
                    },
                    "activation": {
                        "activation_source": "synthetic_flagged",
                        "seed": 20260810,
                        "rows": int(x.shape[0]),
                    },
                },
                scope=codec_scope(tech),
                notes=(f"tensor real extraído do GGUF ({tensor_name}, {tensor_type}); "
                       f"ativação sintética (np.random seed fixo) -> comparison_role=null (§3). "
                       f"{tech} F0={codec_name}+F1 SVD r{int(s.size)}: "
                       f"output cosine={output_q['cosine']:.4f} nrmse={output_q['nrmse']:.4f}; "
                       f"packed {packed_bytes / 1e6:.1f} MB vs fp32 {dense_bytes / 1e6:.1f} MB."))
        except Exception as exc:
            traceback.print_exc()
            recorder.emit(
                f"P1_GGUF_{tech}_CODEC_TENSOR", tech, "FAIL",
                schema_fields=schema,
                scope=codec_scope(tech),
                notes=(f"FAIL do codec {tech} sobre o tensor real {tensor_name}; "
                       "ativação sintética (np.random seed fixo)."),
                error=str(exc))


def codec_scope(tech: str) -> str:
    codec_name = TECH_CODECS[tech][0]
    return (f"codec F0 ({codec_name}) + F1 low-rank SVD da tecnologia {tech} em "
            "NumPy puro sobre UMA Linear REAL extraída do GGUF (dequant fp32 via "
            "pacote gguf pinado); ativação SINTÉTICA FLAGADA (seed fixo) -> "
            "comparison_role=null (§3); quality.output cosine/nrmse REAIS do "
            "tensor real; packed bytes reais; não mede tok/s nem RAM de topo")


# ---------------------------------------------------------------------------
# Limpeza guardada (destrutiva SÓ sob /content|/tmp — nunca em máquina local)
# ---------------------------------------------------------------------------

def cleanup_guarded(paths: List[Path], label: str = "GGUF") -> None:
    allow = _colab_importable() or os.environ.get("RIFT_ALLOW_LOCAL_CLEANUP", "").strip() == "1"
    for path in paths:
        try:
            resolved = Path(path).resolve()
            posix = resolved.as_posix()
            under_safe_root = any(
                posix == root or posix.startswith(root + "/")
                for root in ("/content", "/tmp"))
            if not (allow and under_safe_root):
                print(f"[cleanup:{label}] preservado (fora do Colab//content): {posix}")
                continue
            if resolved.is_dir():
                shutil.rmtree(resolved, ignore_errors=True)
                print(f"[cleanup:{label}] removido: {posix}")
            elif resolved.is_file():
                resolved.unlink(missing_ok=True)
                print(f"[cleanup:{label}] removido: {posix}")
        except Exception as exc:
            print(f"[cleanup:{label}] AVISO: {exc}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def print_final_table(recorder: GgufRecorder) -> None:
    print("\n===== GGUF_RUNTIME_V1 — resumo =====")
    print(f"{'Bateria':<34} {'Tech':<8} {'Status':<8} {'tok/s':>8}")
    print("-" * 64)
    for rec in recorder.records:
        if rec.get("run_id") != recorder.run_id:
            continue
        tok = rec.get("candidate_tok_s")
        tok_txt = f"{tok:8.2f}" if isinstance(tok, (int, float)) else f"{'—':>8}"
        print(f"{rec.get('battery_id', ''):<34} {rec.get('technology', ''):<8} "
              f"{rec.get('status', ''):<8} {tok_txt}")
    print("-" * 64)
    print(f"({BASELINE_NOTE}; registros GGUF nunca entram na política do winner)")


def main(argv: Optional[List[str]] = None) -> int:
    values = list(argv) if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(
        description="Baterias GGUF_RUNTIME_V1 (Muse Glimmer 2-bit no T4 via "
                    "llama.cpp): B0_GGUF_RUNTIME_SETUP, P1_GGUF_E2E_TOKS e "
                    "P1_GGUF_<TECH>_CODEC_TENSOR (docs/C3_CONTRACTS_V1.md §11).")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="repositório GGUF no Hugging Face (default: %(default)s)")
    p.add_argument("--quant", default=DEFAULT_QUANT,
                   help="quant do GGUF a baixar/executar (default: %(default)s)")
    p.add_argument("--out", default="gguf_test_output",
                   help="diretório de saída dos artefatos locais")
    p.add_argument("--publish", default="on", choices=["on", "off"],
                   help="publica cada registro no endpoint de resultados (default: on)")
    p.add_argument("--max-new-tokens", type=int, default=64,
                   help="tokens novos por medição e2e (default: 64)")
    p.add_argument("--ngl", default="auto",
                   help="camadas na GPU (-ngl) ou 'auto' para heurística de "
                        "offload parcial (default: auto)")
    p.add_argument("--skip-codec-tensor", action="store_true",
                   help="pula as 4 baterias P1_GGUF_<TECH>_CODEC_TENSOR")
    p.add_argument("--llama-cpp-ref", default=DEFAULT_LLAMACPP_REF,
                   help="tag PINADA da release oficial ggml-org/llama.cpp "
                        "(default: %(default)s; URL direta via env GGUF_LLAMACPP_URL "
                        "+ checksum via GGUF_LLAMACPP_SHA256)")
    p.add_argument("--server-port", type=int, default=8090,
                   help="porta local do llama-server (default: 8090)")
    p.add_argument("--disk-budget-gb", type=float, default=75.0,
                   help="disco livre mínimo exigido antes dos downloads (default: 75)")
    p.add_argument("--codec-rank", type=int, default=16,
                   help="rank do F1 low-rank SVD nos codec tensors (default: 16)")
    p.add_argument("--codec-max-elements", type=int, default=160_000_000,
                   help="limite de elementos do tensor escolhido (default: 1.6e8)")
    p.add_argument("--results-endpoint", default=None,
                   help="override do endpoint HTTPS de publicação")
    args = p.parse_args(without_ipykernel_connection_args(values))

    bootstrap_colab_secrets()
    hf_token = resolve_hf_token()
    args.model = args.model.strip().replace("https://huggingface.co/", "").strip("/")
    model_id = f"{args.model}:{args.quant.strip()}"
    device_type = "cuda" if nvidia_smi_free_vram_bytes() is not None else "cpu"
    out_dir = Path(args.out)
    run_id = make_run_id()
    recorder = GgufRecorder(
        out_dir, model_id=model_id, run_id=run_id,
        publish_on=args.publish != "off", endpoint=args.results_endpoint)

    print(f"[GGUF] model_id={model_id} device={device_type} "
          f"llama.cpp ref={args.llama_cpp_ref} publish={args.publish}")
    print(f"[GGUF] dependências novas pinadas (homologação TI/SI §11): "
          f"binário llama.cpp {args.llama_cpp_ref} + pip \"{PINNED_GGUF_PIP_SPEC}\"")

    gguf_module = ensure_gguf_module()

    cleanup_paths: List[Path] = [out_dir / "llamacpp" / "archive"]
    try:
        setup = run_setup_battery(args, recorder, model_id, device_type, out_dir, hf_token)
        run_e2e_battery(args, recorder, model_id, device_type, setup, gguf_module)
        if args.skip_codec_tensor:
            print("[codec] --skip-codec-tensor: baterias P1_GGUF_<TECH>_CODEC_TENSOR puladas.")
        else:
            run_codec_tensor_batteries(args, recorder, model_id, device_type,
                                       setup, gguf_module)
        print_final_table(recorder)
        print(f"\n[GGUF] Artefatos locais: {recorder.json_path} | {recorder.csv_path} | "
              f"{recorder.batteries_dir}")
    finally:
        cleanup_guarded(cleanup_paths, label="GGUF")
    return 0


if __name__ == "__main__":
    try:
        _rc = main() or 0
    except SystemExit as _exc:
        _rc = int(_exc.code) if isinstance(_exc.code, int) else 1
    except Exception:
        traceback.print_exc()
        # Não-zero apenas em crash ANTES de qualquer registro gravado.
        _rc = 0 if EMITTED_RECORDS > 0 else 1
    raise SystemExit(_rc)
