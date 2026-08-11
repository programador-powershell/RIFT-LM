#!/usr/bin/env python3
"""C3 — Terceira bateria: metodologia C1 de 16 passos (docs/C3_CONTRACTS_V1.md §2).

Script ÚNICO para as quatro tecnologias:
    python c3_methodology_auto_batteries.py --technology rift|aether|cascade|spectra

Codec F0 default por tecnologia (override com --codec):
    cascade → int4 | rift → int2 | aether → ternary | spectra → ternary
F1 (residual low-rank) e Confidence Gate v0 são comuns a todas.

Mapeamento dos 16 passos → battery_ids (contrato §2):
    1,5   C3_<TECH>_BUNDLE_M0_FREEZE      bundle real + golden tests válidos/inválidos
    2     C3_<TECH>_STAGE_PAGE_M0_FREEZE  stage table ABI (entrada 24B) + golden negativo
    4     C3_<TECH>_IR_WRITER             CASCADE-IR v3 write→validate→reload
    6     C3_<TECH>_CPP_BUNDLE_READER     leitor C++ mmap sobre o bundle real (POSIX; SKIPPED no Windows)
    3,7-12 C3_<TECH>_LINEAR_*             Linear real, 4 caminhos (um registro por caminho)
    11-12 C3_<TECH>_BLOCK_*               Transformer block real, 4 caminhos
    13    publish incremental de cada registro (recorder)
    14    C3_<TECH>_C1_DECISION           aprova/reprova C1 (critérios do contrato)
    15    C3_<TECH>_BLOCKS4_GATED         4 blocos reais patchados (gated vs original)
    16    C3_<TECH>_FULLMODEL_E2E_TOKS    TODOS os blocos patchados; tok/s REAL baseline E candidato
    (15 e 16 só executam com C1_DECISION = PASS; C1 reprovado emite ambos como SKIPPED)

Honestidade de medição (docs/REAL_BENCHMARK_PROTOCOL_V3.md + contrato §3):
  - latência: time.perf_counter_ns com warmup e torch.cuda.synchronize quando CUDA;
  - *_ram_bytes de topo: SOMENTE pico VmRSS medido por fase (thread ~1ms); senão null;
  - candidate_disk_bytes: SOMENTE os.stat de artefatos binários realmente gravados;
  - baseline_tok_s/candidate_tok_s de topo: SOMENTE model.generate do modelo completo;
  - fallback sintético de ativação rebaixa o registro (comparison_role=null + nota).

Auto-contido EXCETO pelo pacote cascade/ (o launcher Colab baixa a lista fixa de
arquivos do pacote); os codecs int2 e ternary do F0 são implementados INLINE aqui.
Sem pip install automático: dependências ausentes geram SystemExit com instruções.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
import threading
import time
import traceback
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# --- resolve pacote cascade/ (cwd do script, /content, /content/cascade_run ou <repo>/core) ---
_HERE = Path(__file__).resolve().parent
for _cand in [_HERE, Path("/content"), Path("/content/cascade_run"), _HERE.parent / "core"]:
    if (_cand / "cascade" / "compiler" / "decompose.py").is_file():
        sys.path.insert(0, str(_cand))
        break

# --- dependências (SEM pip automático: o launcher Colab instala antes) ---
_MISSING_DEPS: List[str] = []
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    _MISSING_DEPS.append("torch")
try:
    import transformers  # noqa: F401  (usado em load_model)
except ImportError:
    _MISSING_DEPS.append("transformers")
if _MISSING_DEPS:
    raise SystemExit(
        "[C3] Dependências ausentes: " + ", ".join(_MISSING_DEPS) + ". "
        "Este script NÃO instala pacotes automaticamente — o launcher Colab deveria "
        "tê-los instalado. Instale manualmente: pip install torch transformers "
        "accelerate sentencepiece (versões pinadas conforme o launcher)."
    )

import cascade  # para localizar cascade/runtime/cpp/mmap_smoke.cpp
from cascade.compiler.block_decompose import find_transformer_blocks
from cascade.compiler.bundle_writer import (
    HEADER_SIZE as CSCD_HEADER_SIZE,
    MAGIC as CSCD_MAGIC,
    VERSION as CSCD_VERSION,
    write_cascade_bundle,
)
from cascade.compiler.cascade_ir import IR_VERSION, make_linear_ir, validate_cascade_ir
from cascade.compiler.decompose import CascadeLinearStages, decompose_linear_int4_lowrank
from cascade.kernels.int4 import dequantize_int4, quantize_int4_group
from cascade.kernels.lowrank import fit_lowrank_residual, lowrank_linear
from cascade.runtime.block_runtime import (
    CascadeLinearModule,
    collect_block_linears,
    restore_block_linears,
)
from cascade.runtime.cleanup import cleanup_colab_workspace
from cascade.runtime.confidence_gate import GateConfig, decide_gate
from cascade.runtime.reference import CascadeLinearRuntime

BENCHMARK_PROTOCOL = "C3_METHODOLOGY_V1"

TECH_DEFAULT_CODEC = {
    "cascade": "int4",
    "rift": "int2",
    "aether": "ternary",
    "spectra": "ternary",
}

# Prompts fixos PT-BR (ativação real da Linear e generate e2e)
ACTIVATION_PROMPT = (
    "Explique por que a memória importa na inferência de modelos de linguagem. "
    "Latência, RAM e disco definem o custo real."
)
GENERATION_PROMPT = "Liste três técnicas para reduzir o uso de memória na inferência de LLMs:"

LATENCY_METHOD = "perf_counter_ns_with_cuda_sync_v1"
RAM_METHOD = "proc_vmrss_sampling_per_phase_v1"


# ---------------------------------------------------------------------------
# Utilidades gerais (espelham cascade_c0/c1/c2)
# ---------------------------------------------------------------------------

def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _pkg_version(name: str) -> Optional[str]:
    try:
        import importlib.metadata
        return importlib.metadata.version(name)
    except Exception:
        return None


def without_ipykernel_connection_args(argv: Iterable[str]) -> List[str]:
    """Remove '-f kernel-*.json' que o ipykernel injeta no Colab (espelha M0)."""
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
    """
    names = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "RIFT_INGEST_TOKEN", "RIFT_RESULTS_ENDPOINT")
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


def ensure_hf_login(token: Optional[str] = None) -> Optional[str]:
    token = token or resolve_hf_token()
    if not token:
        return None
    try:
        from huggingface_hub import login as hf_login
        hf_login(token=token, add_to_git_credential=False)
        print("[auth] HF_TOKEN aplicado.")
    except Exception as exc:
        print(f"[auth] AVISO: {exc}")
    return token


def resolve_device(s: str) -> "torch.device":
    s = (s or "auto").lower()
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def schema_v2_fields(model_id: str, device: "torch.device", codec_name: str) -> Dict[str, Any]:
    """Campos obrigatórios do schema v2 (docs/C3_CONTRACTS_V1.md §3)."""
    torch_v = str(getattr(torch, "__version__", "unknown"))
    raw = f"{BENCHMARK_PROTOCOL}|{model_id}|{device.type}|{torch_v}"
    return {
        "schema_version": 2,
        "benchmark_protocol": BENCHMARK_PROTOCOL,
        "comparison_group_id": "cmp-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24],
        "comparison_context": {
            "protocol": BENCHMARK_PROTOCOL,
            "device": device.type,
            "torch": torch_v,
            "transformers": _pkg_version("transformers"),
            "python": platform.python_version(),
            "codec": codec_name,
        },
        "implementation": {"kind": "REFERENCE_MEASURED", "native": False, "simulated": False},
    }


def _read_vmrss_bytes() -> Optional[int]:
    """VmRSS atual em bytes via /proc/self/status (Linux/Colab); None fora."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


def measure_phase_ram(fn):
    """Executa fn() com thread amostrando VmRSS a ~1ms (RAM real por fase).

    Retorna (resultado_fn, info) onde info = {max_bytes, mean_bytes, n_samples, method}
    ou None quando nenhuma medição real é possível (RAM de topo fica null).
    Fallback getrusage: apenas metrics (pico do processo, não da fase).
    """
    samples: List[int] = []
    stop = threading.Event()

    def _loop():
        while not stop.is_set():
            v = _read_vmrss_bytes()
            if v is not None:
                samples.append(v)
            stop.wait(0.001)

    sampler = threading.Thread(target=_loop, daemon=True)
    sampler.start()
    try:
        result = fn()
    finally:
        stop.set()
        sampler.join(timeout=1.0)
    if samples:
        return result, {
            "max_bytes": int(max(samples)),
            "mean_bytes": int(sum(samples) / len(samples)),
            "n_samples": len(samples),
            "method": RAM_METHOD,
        }
    try:
        import resource
        peak_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if peak_kb > 0:
            return result, {
                "max_bytes": peak_kb * 1024,  # ru_maxrss em KB no Linux
                "mean_bytes": None,
                "n_samples": 0,
                "method": "getrusage_peak_fallback",
            }
    except Exception:
        pass
    return result, None


def ram_top_level(info: Optional[Dict[str, Any]]) -> Optional[int]:
    """RAM de topo: SOMENTE pico VmRSS medido por fase; getrusage é metrics-only."""
    if isinstance(info, dict) and info.get("method") == RAM_METHOD:
        return int(info["max_bytes"])
    return None


def memory_metrics(ram_base, ram_cand, *, estimated_baseline=None, estimated_candidate=None) -> Dict[str, Any]:
    method = None
    if isinstance(ram_cand, dict) and ram_cand.get("method"):
        method = ram_cand["method"]
    elif isinstance(ram_base, dict) and ram_base.get("method"):
        method = ram_base["method"]
    return {
        "method": method,
        "baseline_phase": ram_base,
        "candidate_phase": ram_cand,
        "estimated_baseline_bytes": estimated_baseline,
        "estimated_candidate_bytes": estimated_candidate,
    }


def cosine_nrmse(a: "torch.Tensor", b: "torch.Tensor") -> Dict[str, float]:
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    n = min(a.numel(), b.numel())
    if n == 0:
        return {"cosine": 0.0, "nrmse": 1.0}
    a, b = a[:n], b[:n]
    cos = float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
    denom = float(torch.linalg.vector_norm(a).item()) + 1e-12
    nrmse = float(torch.linalg.vector_norm(a - b).item()) / denom
    return {"cosine": cos, "nrmse": nrmse}


def benchmark_ns(fn, *, warmup: int = 5, iterations: int = 30, device: Optional["torch.device"] = None) -> Dict[str, Any]:
    """Latência real: perf_counter_ns + cuda.synchronize (protocolo V3)."""
    use_cuda = device is not None and getattr(device, "type", "") == "cuda" and torch.cuda.is_available()
    for _ in range(max(0, int(warmup))):
        fn()
    if use_cuda:
        torch.cuda.synchronize()
    times_ms: List[float] = []
    for _ in range(max(1, int(iterations))):
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        fn()
        if use_cuda:
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter_ns() - t0) / 1e6)
    ts = sorted(times_ms)
    n = len(ts)
    mean = sum(ts) / n
    var = sum((t - mean) ** 2 for t in ts) / n
    return {
        "median_ms": ts[n // 2],
        "mean_ms": mean,
        "p95_ms": ts[min(n - 1, int(round(0.95 * (n - 1))))],
        "min_ms": ts[0],
        "max_ms": ts[-1],
        "std_ms": var ** 0.5,
        "iterations": n,
        "warmup": int(warmup),
        "method": LATENCY_METHOD,
    }


def tensor_bytes(t: "torch.Tensor") -> bytes:
    return t.detach().cpu().contiguous().numpy().tobytes()


def stat_bytes(path: Path) -> int:
    return int(os.stat(path).st_size)


# ---------------------------------------------------------------------------
# Codecs F0 (contrato §6) — int4 reutiliza cascade/kernels; int2 e ternary INLINE
# ---------------------------------------------------------------------------

@dataclass
class PackedLinear:
    codec: str
    codes: "torch.Tensor"        # uint8 empacotado
    scales: "torch.Tensor"
    group_size: int
    out_features: int
    in_features: int
    packed_bytes: int            # bytes REAIS (codes + scales)
    meta: Dict[str, Any] = field(default_factory=dict)


def _packed_real_bytes(codes: "torch.Tensor", scales: "torch.Tensor") -> int:
    return int(codes.numel() * codes.element_size() + scales.numel() * scales.element_size())


class _CodecRefAPI:
    """API de referência comum do contrato §6: pack/unpack/linear(x, packed)."""

    def unpack(self, packed: PackedLinear) -> "torch.Tensor":
        """Reconstrói W0 aproximado a partir do PackedLinear (referência)."""
        return self.dequantize(
            packed.codes, packed.scales, group_size=packed.group_size,
            out_features=packed.out_features, in_features=packed.in_features,
        )

    def linear(self, x: "torch.Tensor", packed: PackedLinear) -> "torch.Tensor":
        """F0 de referência: Y = X @ unpack(packed)^T."""
        return F.linear(x.to(dtype=torch.float32), self.unpack(packed))


class Int4Codec(_CodecRefAPI):
    """F0 INT4 groupwise — reutiliza cascade/kernels/int4.py."""
    name = "int4"

    def pack(self, w: "torch.Tensor", *, group_size: int) -> PackedLinear:
        out_f, in_f = w.shape
        codes, scales, gs = quantize_int4_group(w, group_size=group_size)
        return PackedLinear(
            codec=self.name, codes=codes, scales=scales, group_size=int(gs),
            out_features=int(out_f), in_features=int(in_f),
            packed_bytes=_packed_real_bytes(codes, scales),
            meta={"levels": 16, "scheme": "groupwise_signed_int4"},
        )

    def dequantize(self, codes, scales, *, group_size, out_features, in_features):
        return dequantize_int4(
            codes, scales, group_size=group_size,
            out_features=out_features, in_features=in_features,
        )


class Int2Codec(_CodecRefAPI):
    """F0 INT2 (RIFT): groupwise, 2 bits/peso, 4 níveis simétricos {-3,-1,+1,+3}·(escala/3)."""
    name = "int2"

    def pack(self, w: "torch.Tensor", *, group_size: int) -> PackedLinear:
        out_f, in_f = w.shape
        gs = max(4, int(group_size))
        w2 = w
        if in_f % gs != 0:
            w2 = F.pad(w2, (0, gs - (in_f % gs)))
        in_p = w2.shape[1]
        n_groups = in_p // gs
        wg = w2.view(out_f, n_groups, gs)
        scales = (wg.abs().amax(dim=2).clamp_min(1e-8)).to(torch.float32).contiguous()
        unit = (scales / 3.0)[:, :, None]
        q = torch.round((wg / unit + 3.0) / 2.0).clamp(0, 3).to(torch.int16).view(out_f, in_p)
        if in_p % 4 != 0:
            q = F.pad(q, (0, 4 - (in_p % 4)))
        codes = (
            q[:, 0::4] | (q[:, 1::4] << 2) | (q[:, 2::4] << 4) | (q[:, 3::4] << 6)
        ).to(torch.uint8).contiguous()
        return PackedLinear(
            codec=self.name, codes=codes, scales=scales, group_size=gs,
            out_features=int(out_f), in_features=int(in_f),
            packed_bytes=_packed_real_bytes(codes, scales),
            meta={"levels": 4, "scheme": "groupwise_symmetric_int2"},
        )

    def dequantize(self, codes, scales, *, group_size, out_features, in_features):
        gs = int(group_size)
        c = codes.to(torch.int16)
        dev = codes.device
        decoded = torch.empty(c.shape[0], c.shape[1] * 4, dtype=torch.float32, device=dev)
        decoded[:, 0::4] = (c & 0x03).to(torch.float32)
        decoded[:, 1::4] = ((c >> 2) & 0x03).to(torch.float32)
        decoded[:, 2::4] = ((c >> 4) & 0x03).to(torch.float32)
        decoded[:, 3::4] = ((c >> 6) & 0x03).to(torch.float32)
        levels = decoded * 2.0 - 3.0  # índices 0..3 → {-3,-1,+1,+3}
        n_groups = int(scales.shape[1])
        in_p = n_groups * gs
        levels = levels[:, :in_p]
        sc = scales.to(device=dev, dtype=torch.float32)
        w = levels.view(out_features, n_groups, gs) * (sc / 3.0)[:, :, None]
        return w.reshape(out_features, in_p)[:, :in_features].contiguous()


class TernaryCodec(_CodecRefAPI):
    """F0 ternário (AETHER/SPECTRA): {-1,0,+1}, escala por linha + limiar de esparsidade."""
    name = "ternary"
    threshold_ratio = 0.7  # limiar clássico TWN: 0.7 · mean(|w|) por linha

    def pack(self, w: "torch.Tensor", *, group_size: int) -> PackedLinear:
        out_f, in_f = w.shape
        absw = w.abs()
        thr = (self.threshold_ratio * absw.mean(dim=1, keepdim=True)).clamp_min(1e-12)
        mask = absw > thr
        denom = mask.sum(dim=1).clamp_min(1)
        scales = ((absw * mask).sum(dim=1) / denom).clamp_min(1e-8).to(torch.float32).contiguous()
        idx = torch.zeros_like(w, dtype=torch.int16)
        idx[w > thr] = 1    # +1
        idx[w < -thr] = 2   # -1
        sparsity = float((idx == 0).float().mean().item())
        if in_f % 4 != 0:
            idx = F.pad(idx, (0, 4 - (in_f % 4)))
        codes = (
            idx[:, 0::4] | (idx[:, 1::4] << 2) | (idx[:, 2::4] << 4) | (idx[:, 3::4] << 6)
        ).to(torch.uint8).contiguous()
        return PackedLinear(
            codec=self.name, codes=codes, scales=scales, group_size=int(group_size),
            out_features=int(out_f), in_features=int(in_f),
            packed_bytes=_packed_real_bytes(codes, scales),
            meta={
                "levels": 3, "scheme": "ternary_rowscale_threshold",
                "threshold_ratio": self.threshold_ratio, "sparsity": sparsity,
            },
        )

    def dequantize(self, codes, scales, *, group_size, out_features, in_features):
        c = codes.to(torch.int16)
        dev = codes.device
        decoded = torch.empty(c.shape[0], c.shape[1] * 4, dtype=torch.int16, device=dev)
        decoded[:, 0::4] = c & 0x03
        decoded[:, 1::4] = (c >> 2) & 0x03
        decoded[:, 2::4] = (c >> 4) & 0x03
        decoded[:, 3::4] = (c >> 6) & 0x03
        val = torch.zeros_like(decoded, dtype=torch.float32)
        val[decoded == 1] = 1.0
        val[decoded == 2] = -1.0
        sc = scales.to(device=dev, dtype=torch.float32).view(-1, 1)
        return (val[:, :in_features] * sc).contiguous()


_CODECS = {"int4": Int4Codec, "int2": Int2Codec, "ternary": TernaryCodec}

# Rótulo do stage.meta.codec do stage 0 no bundle CSCD (contrato §6:
# "Bundles gravam stage.meta.codec" — precisa declarar o codec REAL do payload).
F0_BUNDLE_CODEC_LABEL = {
    "int4": "INT4_GROUP",
    "int2": "INT2_GROUP",
    "ternary": "TERNARY_ROWSCALE",
}


def get_codec(name: str):
    if name not in _CODECS:
        raise SystemExit(f"[C3] codec desconhecido: {name} (válidos: {sorted(_CODECS)})")
    return _CODECS[name]()


def decompose_with_codec(weight: "torch.Tensor", codec, *, rank: int, group_size: int) -> Tuple[CascadeLinearStages, PackedLinear]:
    """W ≈ F0(codec) + U·diag(S)·Vᵀ. Para int4 reutiliza decompose_linear_int4_lowrank."""
    w = weight.detach().to(dtype=torch.float32).cpu().contiguous()
    if codec.name == "int4":
        stages = decompose_linear_int4_lowrank(w, rank=rank, group_size=group_size)
        packed = PackedLinear(
            codec="int4", codes=stages.codes, scales=stages.scales,
            group_size=stages.group_size, out_features=stages.out_features,
            in_features=stages.in_features,
            packed_bytes=_packed_real_bytes(stages.codes, stages.scales),
            meta={"levels": 16, "scheme": "groupwise_signed_int4"},
        )
        return stages, packed
    out_f, in_f = w.shape
    packed = codec.pack(w, group_size=group_size)
    w0 = codec.dequantize(
        packed.codes, packed.scales, group_size=packed.group_size,
        out_features=out_f, in_features=in_f,
    )
    residual = w - w0
    u, s, v = fit_lowrank_residual(residual, rank=rank)
    f1_bytes = int((u.numel() + s.numel() + v.numel()) * 4)
    stages = CascadeLinearStages(
        out_features=out_f,
        in_features=in_f,
        group_size=packed.group_size,
        codes=packed.codes,
        scales=packed.scales,
        u=u,
        s=s,
        v=v,
        rank=int(s.numel()),
        f0_bytes=packed.packed_bytes,
        f1_bytes=f1_bytes,
        baseline_bytes=int(w.numel() * 4),
    )
    return stages, packed


def write_stage_artifacts(artifacts_dir: Path, prefix: str, stages: CascadeLinearStages) -> Dict[str, Any]:
    """Grava payloads F0/F1 REAIS em disco e retorna bytes via os.stat (protocolo V3)."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    f0_path = artifacts_dir / f"{prefix}_f0.bin"
    f1_path = artifacts_dir / f"{prefix}_f1.bin"
    f0_path.write_bytes(tensor_bytes(stages.codes) + tensor_bytes(stages.scales))
    f1_path.write_bytes(tensor_bytes(stages.u) + tensor_bytes(stages.s) + tensor_bytes(stages.v))
    f0_b = stat_bytes(f0_path)
    f1_b = stat_bytes(f1_path)
    return {
        "f0_path": str(f0_path), "f1_path": str(f1_path),
        "f0_bytes": f0_b, "f1_bytes": f1_b, "total_bytes": f0_b + f1_b,
        "method": "binary_os_stat_v1",
    }

# ---------------------------------------------------------------------------
# Leitor/validador do bundle CSCD (golden tests M0 — passos 1, 2 e 5)
# ---------------------------------------------------------------------------

# layout do header (cascade/compiler/bundle_writer.py):
# magic(4s) ver(H) flags(H) hdr(I) n_stages(I) ir(Q) st(Q) gate(Q) pay(Q) fsize(Q) crc(Q)
CSCD_HEADER_PREFIX_FMT = "<4sHHIIQQQQQQ"
CSCD_CRC_OFFSET = struct.calcsize("<4sHHIIQQQQQ")  # 56: crc é o 6º Q
CSCD_STAGE_ENTRY_FMT = "<QQII"
CSCD_STAGE_ENTRY_SIZE = struct.calcsize(CSCD_STAGE_ENTRY_FMT)  # 24 bytes (ABI congelada)


class BundleFormatError(ValueError):
    """Bundle CSCD inválido/corrompido — o leitor validador rejeita com motivo."""


def validate_cscd_bundle(buf: bytes) -> Dict[str, Any]:
    """Leitor validador: header, tamanhos, CRC e limites de cada stage entry.

    Levanta BundleFormatError com motivo curto; retorna campos parseados quando válido.
    """
    if len(buf) < CSCD_HEADER_SIZE:
        raise BundleFormatError("truncated_header")
    (
        magic, version, _flags, hdr_size, n_stages,
        ir_off, st_off, gate_off, pay_off, fsize, crc,
    ) = struct.unpack_from(CSCD_HEADER_PREFIX_FMT, buf, 0)
    if magic != CSCD_MAGIC:
        raise BundleFormatError("bad_magic")
    if version != CSCD_VERSION:
        raise BundleFormatError("bad_version")
    if hdr_size != CSCD_HEADER_SIZE:
        raise BundleFormatError("bad_header_size")
    if fsize != len(buf):
        raise BundleFormatError("bad_file_size")
    if (zlib.crc32(buf[CSCD_HEADER_SIZE:]) & 0xFFFFFFFFFFFFFFFF) != crc:
        raise BundleFormatError("bad_crc")
    if not (CSCD_HEADER_SIZE <= ir_off < len(buf)):
        raise BundleFormatError("ir_offset_out_of_bounds")
    if st_off + n_stages * CSCD_STAGE_ENTRY_SIZE > len(buf):
        raise BundleFormatError("stage_table_out_of_bounds")
    stages: List[Dict[str, int]] = []
    for i in range(int(n_stages)):
        off, sz, sid, sflags = struct.unpack_from(CSCD_STAGE_ENTRY_FMT, buf, int(st_off) + i * CSCD_STAGE_ENTRY_SIZE)
        if off < CSCD_HEADER_SIZE or off + sz > len(buf):
            raise BundleFormatError(f"stage_{i}_payload_out_of_bounds")
        if sz <= 4:
            raise BundleFormatError(f"stage_{i}_payload_too_small")
        stages.append({"offset": int(off), "size": int(sz), "stage_id": int(sid), "flags": int(sflags)})
    return {
        "version": int(version), "n_stages": int(n_stages),
        "ir_offset": int(ir_off), "stage_table_offset": int(st_off),
        "gate_table_offset": int(gate_off), "payload_offset": int(pay_off),
        "file_size": int(fsize), "checksum": int(crc), "stages": stages,
    }


def _refix_crc(data: bytes) -> bytes:
    """Recalcula o CRC do corpo e regrava no header (para negativos direcionados)."""
    crc = zlib.crc32(data[CSCD_HEADER_SIZE:]) & 0xFFFFFFFFFFFFFFFF
    return data[:CSCD_CRC_OFFSET] + struct.pack("<Q", crc) + data[CSCD_CRC_OFFSET + 8:]


def run_bundle_golden_tests(buf: bytes) -> Dict[str, Any]:
    """Golden tests do bundle M0: caso válido PASS + 5 negativos REJEITADOS."""
    results: Dict[str, Any] = {}
    try:
        parsed = validate_cscd_bundle(buf)
        results["valid_bundle"] = {"ok": True, "n_stages": parsed["n_stages"], "version": parsed["version"]}
    except BundleFormatError as exc:
        results["valid_bundle"] = {"ok": False, "error": str(exc)}
        parsed = None

    def _expect_reject(name: str, corrupted: bytes, expected: Optional[str] = None) -> None:
        try:
            validate_cscd_bundle(corrupted)
            results[name] = {"rejected": False, "error": None}
        except BundleFormatError as exc:
            ok_reason = expected is None or str(exc) == expected
            results[name] = {"rejected": True, "error": str(exc), "expected_reason_ok": ok_reason}

    # 1) magic inválido
    _expect_reject("bad_magic", b"XXXX" + buf[4:], "bad_magic")
    # 2) versão inválida
    bad_ver = bytearray(buf)
    struct.pack_into("<H", bad_ver, 4, 0x0009)
    _expect_reject("bad_version", bytes(bad_ver), "bad_version")
    # 3) header truncado
    _expect_reject("truncated_header", buf[:100], "truncated_header")
    # 4) CRC corrompido (flip de 1 byte do corpo, sem recalcular o CRC)
    bad_crc = bytearray(buf)
    bad_crc[-1] ^= 0xFF
    _expect_reject("corrupted_crc", bytes(bad_crc), "bad_crc")
    # 5) stage offset fora dos limites (CRC recalculado para isolar o teste de bounds)
    if parsed is not None and parsed["stages"]:
        oob = bytearray(buf)
        st_off = parsed["stage_table_offset"]
        struct.pack_into("<Q", oob, st_off, len(buf) + 4096)
        _expect_reject("stage_offset_out_of_bounds", _refix_crc(bytes(oob)), "stage_0_payload_out_of_bounds")
    else:
        results["stage_offset_out_of_bounds"] = {"rejected": False, "error": "bundle válido indisponível"}

    negatives = [k for k in results if k != "valid_bundle"]
    all_ok = bool(results["valid_bundle"].get("ok")) and all(
        results[k].get("rejected") for k in negatives
    )
    results["golden_pass"] = all_ok
    return results


def _read_stage_meta(buf: bytes, off: int) -> Dict[str, Any]:
    (mlen,) = struct.unpack_from("<I", buf, off)
    return json.loads(buf[off + 4: off + 4 + mlen].decode("utf-8"))


def run_stage_page_checks(buf: bytes, *, expected_f0_codec: Optional[str] = None) -> Dict[str, Any]:
    """Passo 2: ABI da stage table (entrada de 24 bytes) — contagem/ordem/alinhamento/tipos.

    Quando `expected_f0_codec` é dado, também verifica que o stage.meta.codec do
    stage 0 declara o codec REAL da tecnologia (falha o ABI check se divergir).
    """
    parsed = validate_cscd_bundle(buf)
    stages = parsed["stages"]
    checks: Dict[str, Any] = {"entry_size_bytes": CSCD_STAGE_ENTRY_SIZE}
    checks["entry_size_ok"] = CSCD_STAGE_ENTRY_SIZE == 24
    checks["count_ok"] = len(stages) == parsed["n_stages"] == 2
    offsets = [s["offset"] for s in stages]
    checks["order_ok"] = offsets == sorted(offsets) and len(set(offsets)) == len(offsets)
    checks["alignment_ok"] = all(s["offset"] % 64 == 0 for s in stages)
    checks["ids_ok"] = [s["stage_id"] for s in stages] == list(range(len(stages)))
    checks["bounds_ok"] = all(
        s["offset"] >= parsed["payload_offset"] and s["offset"] + s["size"] <= parsed["file_size"]
        for s in stages
    )
    metas = []
    try:
        for i, s in enumerate(stages):
            meta = _read_stage_meta(buf, s["offset"])
            metas.append({k: meta.get(k) for k in ("stage_id", "stage_index", "stage_type", "codec", "rank", "group_size")})
        checks["types_ok"] = (
            metas[0].get("stage_type") == "BASE_STAGE"
            and metas[1].get("stage_type") == "RESIDUAL_LOWRANK"
            and metas[0].get("stage_index") == 0
            and metas[1].get("stage_index") == 1
        )
    except Exception as exc:
        checks["types_ok"] = False
        checks["meta_error"] = str(exc)[:200]
    checks["stage_metas"] = metas
    if expected_f0_codec is not None:
        declared = metas[0].get("codec") if metas else None
        checks["f0_codec_declared"] = declared
        checks["f0_codec_expected"] = expected_f0_codec
        checks["f0_codec_ok"] = declared == expected_f0_codec

    # golden negativo: corromper o size da stage 0 (CRC recalculado) → rejeição
    bad = bytearray(buf)
    struct.pack_into("<Q", bad, parsed["stage_table_offset"] + 8, len(buf) * 2)
    try:
        validate_cscd_bundle(_refix_crc(bytes(bad)))
        checks["negative_corrupt_size_rejected"] = False
    except BundleFormatError as exc:
        checks["negative_corrupt_size_rejected"] = True
        checks["negative_error"] = str(exc)

    pass_keys = [
        "entry_size_ok", "count_ok", "order_ok", "alignment_ok",
        "ids_ok", "bounds_ok", "types_ok", "negative_corrupt_size_rejected",
    ]
    if expected_f0_codec is not None:
        pass_keys.append("f0_codec_ok")
    checks["pass"] = all(bool(checks.get(k)) for k in pass_keys)
    return checks


def run_ir_roundtrip(out_dir: Path, model_id: str, target_layer: str) -> Dict[str, Any]:
    """Passo 4: CASCADE-IR v3 write→validate→reload roundtrip sobre a Linear real."""
    ir = make_linear_ir(
        model_id=model_id, architecture_hint="dense-linear",
        target_layer=target_layer, cascade_ref=0,
    )
    validate_cascade_ir(ir)
    ir_path = out_dir / "c3_linear_ir.json"
    ir_path.write_text(json.dumps(ir, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    reloaded = json.loads(ir_path.read_text(encoding="utf-8"))
    validate_cascade_ir(reloaded)
    equal = reloaded == ir
    return {
        "ir_version": IR_VERSION,
        "path": str(ir_path),
        "roundtrip_equal": bool(equal),
        "n_operations": len(ir["operations"]),
        "pass": bool(equal),
    }


def run_cpp_bundle_reader(bundle_path: Path, out_dir: Path) -> Dict[str, Any]:
    """Passo 6: compila e executa o leitor C++ mmap sobre o bundle real (POSIX).

    Windows/sem compilador → status SKIPPED. Quando executa de fato:
    implementation.kind = NATIVE_MEASURED e wall-clock real do read+validate.
    """
    info: Dict[str, Any] = {"status": "SKIPPED", "ran": False}
    if os.name != "posix":
        info["note"] = "POSIX-only (mmap): indisponível no Windows — SKIPPED por contrato"
        return info
    compiler = shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")
    has_cmake = shutil.which("cmake") is not None
    if compiler is None:
        info["note"] = (
            "Nenhum compilador C++ (g++/c++/clang++) no PATH"
            + (" (cmake presente, mas sem toolchain)" if has_cmake else "")
            + " — SKIPPED"
        )
        return info
    cpp_dir = Path(cascade.__file__).resolve().parent / "runtime" / "cpp"
    src = cpp_dir / "mmap_smoke.cpp"
    if not src.is_file():
        info["note"] = f"Fonte não encontrada: {src} — SKIPPED"
        return info
    exe = out_dir / "cascade_mmap_smoke"
    t0 = time.perf_counter_ns()
    try:
        comp = subprocess.run(
            [compiler, "-std=c++20", "-O2", "-I", str(cpp_dir), str(src), "-o", str(exe)],
            capture_output=True, text=True, timeout=300,
        )
    except Exception as exc:
        info["status"] = "FAIL"
        info["note"] = f"Falha ao invocar compilador: {exc}"
        return info
    compile_ms = (time.perf_counter_ns() - t0) / 1e6
    info["compiler"] = compiler
    info["compile_ms"] = compile_ms
    if comp.returncode != 0:
        info["status"] = "FAIL"
        info["note"] = ("Compilação falhou: " + (comp.stderr or "")[:400])
        return info
    t1 = time.perf_counter_ns()
    try:
        run = subprocess.run([str(exe), str(bundle_path)], capture_output=True, text=True, timeout=120)
    except Exception as exc:
        info["status"] = "FAIL"
        info["note"] = f"Falha ao executar leitor: {exc}"
        return info
    run_ms = (time.perf_counter_ns() - t1) / 1e6
    info["ran"] = True
    info["read_validate_ms"] = run_ms
    info["exit_code"] = int(run.returncode)
    info["stdout"] = (run.stdout or "")[:400]
    info["stderr"] = (run.stderr or "")[:400]
    info["status"] = "PASS" if run.returncode == 0 else "FAIL"
    info["note"] = (
        f"Leitor C++ mmap real: exit={run.returncode} read+validate={run_ms:.2f} ms"
    )
    return info


# ---------------------------------------------------------------------------
# Runtimes C3 — Linear (4 caminhos) e módulo de bloco para codecs inline
# ---------------------------------------------------------------------------

class C3LinearRuntime:
    """Linear de referência com 4 caminhos; para int4 delega a CascadeLinearRuntime."""

    def __init__(self, stages: CascadeLinearStages, packed: PackedLinear, codec, *, gate_percentile: float = 70.0):
        self.stages = stages
        self.packed = packed
        self.codec = codec
        self.gate_cfg = GateConfig(percentile=gate_percentile)
        self._ref: Optional[CascadeLinearRuntime] = None
        if codec.name == "int4":
            self._ref = CascadeLinearRuntime(
                stages, gate_percentile=gate_percentile, device=torch.device("cpu")
            )

    def execute(self, x: "torch.Tensor", *, path: str) -> Dict[str, Any]:
        if self._ref is not None:
            r = self._ref.execute(x, path=path)
            m = r["metrics"]
            return {
                "y": r["y"], "f0_calls": int(m.f0_calls), "f1_calls": int(m.f1_calls),
                "f1_skip_rate": float(m.f1_skip_rate), "gate": r.get("gate"),
            }
        x = x.to(dtype=torch.float32)
        st = self.stages
        # F0 dequantizado por chamada (espelha o caminho de referência int4)
        w0 = self.codec.dequantize(
            self.packed.codes, self.packed.scales, group_size=self.packed.group_size,
            out_features=st.out_features, in_features=st.in_features,
        )
        y0 = F.linear(x, w0)
        n = int(x.shape[0])
        if path == "F0_ONLY":
            return {"y": y0, "f0_calls": n, "f1_calls": 0, "f1_skip_rate": 1.0, "gate": None}
        y1 = lowrank_linear(x, st.u, st.s, st.v)
        if path == "F0_PLUS_F1_ALWAYS":
            return {"y": y0 + y1, "f0_calls": n, "f1_calls": n, "f1_skip_rate": 0.0, "gate": None}
        # F0_GATE_F1
        mask, gate_meta = decide_gate(x, self.gate_cfg)
        y = y0 + mask.to(dtype=y1.dtype).unsqueeze(1) * y1
        f1_calls = int(mask.to(torch.int64).sum().item())
        return {
            "y": y, "f0_calls": n, "f1_calls": f1_calls,
            "f1_skip_rate": 1.0 - (f1_calls / max(n, 1)), "gate": gate_meta,
        }


class C3InlineLinearModule(nn.Module):
    """Substitui nn.Linear com F0 do codec inline (int2/ternary) + Gate·F1.

    Espelha cascade/runtime/block_runtime.CascadeLinearModule (mesmos contadores
    f0_calls/f1_calls/f1_skip_calls, path, stats()) — o W denso original NÃO fica
    no caminho quente (apenas codes/scales/u/s/v residem no módulo).
    """

    def __init__(self, stages: CascadeLinearStages, codec, *, gate_percentile: float = 70.0,
                 path: str = "F0_GATE_F1", low_mem: Optional[bool] = None):
        super().__init__()
        self.codec = codec
        self.path = path
        self.gate_cfg = GateConfig(percentile=gate_percentile)
        self.register_buffer("codes", stages.codes)
        self.register_buffer("scales", stages.scales)
        self.register_buffer("u", stages.u)
        self.register_buffer("s", stages.s)
        self.register_buffer("v", stages.v)
        self.group_size = int(stages.group_size)
        self.out_features = int(stages.out_features)
        self.in_features = int(stages.in_features)
        self.bias = None
        self.last_gate_rate = 1.0
        self.f0_calls = 0
        self.f1_calls = 0
        self.f1_skip_calls = 0
        if low_mem is None:
            low_mem = os.environ.get("CASCADE_LOW_MEM", "").strip() == "1"
        self.low_mem = bool(low_mem)
        self._w0_cache: Optional["torch.Tensor"] = None

    def _dequant_w0(self, device: "torch.device") -> "torch.Tensor":
        w0 = self.codec.dequantize(
            self.codes, self.scales, group_size=self.group_size,
            out_features=self.out_features, in_features=self.in_features,
        )
        return w0.to(device=device, dtype=torch.float32)

    def _w0(self, device: "torch.device") -> "torch.Tensor":
        if self.low_mem:
            return self._dequant_w0(device)
        if self._w0_cache is None or self._w0_cache.device != device:
            self._w0_cache = self._dequant_w0(device)
        return self._w0_cache

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1])
        x2f = x2.to(dtype=torch.float32)
        w0 = self._w0(x2f.device)
        y0 = F.linear(x2f, w0)
        n = int(x2f.shape[0])
        self.f0_calls += n

        if self.path == "F0_ONLY":
            self.last_gate_rate = 0.0
            self.f1_skip_calls += n
            out = y0
        else:
            y1 = lowrank_linear(
                x2f,
                self.u.to(device=x2f.device, dtype=torch.float32),
                self.s.to(device=x2f.device, dtype=torch.float32),
                self.v.to(device=x2f.device, dtype=torch.float32),
            )
            if self.path == "F0_PLUS_F1_ALWAYS":
                self.f1_calls += n
                self.last_gate_rate = 1.0
                out = y0 + y1
            else:
                mask, meta = decide_gate(x2f, self.gate_cfg)
                self.last_gate_rate = float(meta["activation_rate"])
                applied = int(mask.to(torch.int64).sum().item())
                self.f1_calls += applied
                self.f1_skip_calls += n - applied
                out = y0 + mask.to(dtype=y1.dtype).unsqueeze(1) * y1

        if x.dtype != out.dtype:
            out = out.to(dtype=x.dtype)
        return out.reshape(*orig_shape[:-1], self.out_features)

    def stats(self) -> Dict[str, Any]:
        total = max(self.f0_calls, 1)
        resident = int(
            self.codes.numel() * self.codes.element_size()
            + self.scales.numel() * self.scales.element_size()
            + self.u.numel() * 4
            + self.s.numel() * 4
            + self.v.numel() * 4
        )
        w0_cache_bytes = (
            int(self._w0_cache.numel() * self._w0_cache.element_size())
            if self._w0_cache is not None else 0
        )
        return {
            "f0_calls": self.f0_calls,
            "f1_calls": self.f1_calls,
            "f1_skip_rate": 1.0 - (self.f1_calls / total),
            "last_gate_rate": self.last_gate_rate,
            "resident_bytes": resident,
            "w0_cache_bytes": w0_cache_bytes,
            "resident_bytes_with_cache": resident + w0_cache_bytes,
            "low_mem": self.low_mem,
        }


def set_module_by_path(root: nn.Module, dotted: str, new_mod: nn.Module) -> None:
    """Substitui submódulo por caminho pontilhado (suporta índices numéricos)."""
    parts = dotted.split(".") if dotted else []
    if not parts:
        raise ValueError("caminho de módulo vazio")
    parent = root
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    leaf = parts[-1]
    if leaf.isdigit():
        parent[int(leaf)] = new_mod
    else:
        setattr(parent, leaf, new_mod)


def patch_one_block(
    block: nn.Module,
    block_name: str,
    codec,
    *,
    rank: int,
    group_size: int,
    gate_percentile: float,
    path: str,
    device: Optional["torch.device"],
    decomp_cache: Dict[str, Tuple[CascadeLinearStages, PackedLinear]],
    artifacts_dir: Optional[Path],
    artifact_registry: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, nn.Linear], Dict[str, nn.Module]]:
    """Troca TODAS as nn.Linear do bloco pelo runtime do codec (W fora do caminho quente).

    Reutiliza CascadeLinearModule para int4; para int2/ternary usa o módulo inline.
    Decomposições são cacheadas por nome (reuso entre passos 11-12, 15 e 16) e os
    payloads F0/F1 são gravados UMA vez como artefatos binários reais.

    TRANSACIONAL: se qualquer Linear falhar no meio do loop (ex.: OOM em .to(device),
    falha do pca_lowrank), as Linears já substituídas são RESTAURADAS a partir de
    `originals` antes de relançar — o bloco nunca fica parcialmente patchado.
    """
    originals = collect_block_linears(block, block_name)
    replaced: Dict[str, nn.Module] = {}
    try:
        for full, linear in originals.items():
            if full in decomp_cache:
                stages, packed = decomp_cache[full]
            else:
                w = linear.weight.detach().float().cpu()
                stages, packed = decompose_with_codec(w, codec, rank=rank, group_size=group_size)
                decomp_cache[full] = (stages, packed)
            if artifacts_dir is not None and full not in artifact_registry:
                artifact_registry[full] = write_stage_artifacts(
                    artifacts_dir, full.replace(".", "_"), stages
                )
            if codec.name == "int4":
                mod: nn.Module = CascadeLinearModule(stages, gate_percentile=gate_percentile, path=path)
            else:
                mod = C3InlineLinearModule(stages, codec, gate_percentile=gate_percentile, path=path)
            if device is not None:
                mod = mod.to(device)
            short = full[len(block_name) + 1:] if full.startswith(block_name + ".") else full
            set_module_by_path(block, short, mod)
            replaced[full] = mod
    except Exception:
        # rollback: devolve as Linears originais já substituídas e relança
        for full in list(replaced):
            short = full[len(block_name) + 1:] if full.startswith(block_name + ".") else full
            try:
                set_module_by_path(block, short, originals[full])
            except Exception as rexc:
                print(f"[C3] AVISO rollback de {full} falhou: {rexc}")
        raise
    if not replaced:
        raise RuntimeError(f"Nenhuma nn.Linear em {block_name}")
    return originals, replaced


def set_replaced_path(replaced: Dict[str, nn.Module], path: str) -> None:
    for m in replaced.values():
        m.path = path


def reset_replaced_counters(replaced: Dict[str, nn.Module]) -> None:
    """Zera contadores ANTES de cada caminho (evita o bug de fechamento do C1)."""
    for m in replaced.values():
        m.f0_calls = 0
        m.f1_calls = 0
        m.f1_skip_calls = 0


def read_replaced_counters(replaced: Dict[str, nn.Module]) -> Dict[str, Any]:
    """Lê contadores DEPOIS do caminho executado (por caminho, nunca acumulado)."""
    f0 = sum(int(m.f0_calls) for m in replaced.values())
    f1 = sum(int(m.f1_calls) for m in replaced.values())
    return {
        "F0_calls": f0,
        "F1_calls": f1,
        "F1_skip_rate": 1.0 - (f1 / max(f0, 1)),
        "avg_stages_per_token": 1.0 + (f1 / max(f0, 1)),
    }

# ---------------------------------------------------------------------------
# Modelo real (HF) — load, Linear alvo, ativação real, blocos e generate
# ---------------------------------------------------------------------------

def load_model(model_id: str, device: "torch.device", trust: bool, token: Optional[str]):
    try:
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "[C3] transformers indisponível. Instale manualmente: pip install "
            f"transformers accelerate sentencepiece. Erro: {exc}"
        )
    try:
        from transformers import AutoModelForMultimodalLM  # type: ignore
    except Exception:
        AutoModelForMultimodalLM = None  # type: ignore
    tok = AutoTokenizer.from_pretrained(model_id, token=token, trust_remote_code=trust)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    kwargs = dict(
        token=token, trust_remote_code=trust, low_cpu_mem_usage=True,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    if device.type == "cuda":
        kwargs["device_map"] = "auto"
    classes = []
    if AutoModelForMultimodalLM is not None:
        classes.append(AutoModelForMultimodalLM)
    classes += [AutoModelForCausalLM, AutoModel]
    errors = []
    model = None
    for cls in classes:
        try:
            model = cls.from_pretrained(model_id, **kwargs)
            if getattr(model, "hf_device_map", None) is None:
                model = model.to(device)
            model.eval()
            print(f"[C3] load {cls.__name__}")
            break
        except Exception as exc:
            errors.append(f"{cls.__name__}: {exc}")
    if model is None:
        raise RuntimeError("Falha ao carregar modelo:\n" + "\n".join(errors))
    return model, tok


def find_linear_weight(model, target: str = "auto") -> Tuple[str, "torch.Tensor"]:
    """Seleciona a Linear real alvo (espelha cascade_c0)."""
    state = model.state_dict()
    if target and target != "auto" and target in state:
        return target, state[target].detach().float()
    if target and target != "auto" and not target.endswith(".weight"):
        alt = target + ".weight"
        if alt in state:
            return alt, state[alt].detach().float()
    for name, tensor in state.items():
        if tensor.ndim == 2 and "embed" not in name.lower() and tensor.shape[0] > 32 and tensor.shape[1] > 32:
            if any(k in name for k in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "fc", "dense", "linear")):
                return name, tensor.detach().float()
    for name, tensor in state.items():
        if tensor.ndim == 2 and min(tensor.shape) >= 64:
            return name, tensor.detach().float()
    raise RuntimeError("Nenhuma Linear 2D adequada encontrada no state_dict")


def resolve_module_by_name(model: nn.Module, dotted: str) -> nn.Module:
    mod = model
    for part in dotted.replace(".weight", "").split("."):
        mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
    return mod


def capture_activation(model, tokenizer, layer_name: str, device: "torch.device", prompt: str) -> Optional["torch.Tensor"]:
    """Captura a entrada REAL da Linear alvo via forward hook; None se falhar."""
    captured: Dict[str, "torch.Tensor"] = {}

    def hook(_mod, inputs, _output):
        if inputs and torch.is_tensor(inputs[0]):
            captured["x"] = inputs[0].detach()

    try:
        mod = resolve_module_by_name(model, layer_name)
        handle = mod.register_forward_hook(hook)
        try:
            enc = tokenizer(prompt, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.inference_mode():
                model(**enc)
        finally:
            handle.remove()
        if "x" in captured:
            x = captured["x"].float().cpu()
            if x.ndim >= 3:
                x = x.reshape(-1, x.shape[-1])
            elif x.ndim == 1:
                x = x.unsqueeze(0)
            return x.contiguous()
    except Exception as exc:
        print(f"[C3] AVISO captura de ativação: {exc}")
    return None


_BLOCK_CACHE_KWARGS = ("past_key_value", "past_key_values", "layer_past")


def capture_block_io(model: nn.Module, block: nn.Module, inputs: Dict[str, "torch.Tensor"]) -> Dict[str, Any]:
    """Captura entradas REAIS (args/kwargs) e saída do bloco em um forward do modelo."""
    captured: Dict[str, Any] = {}

    def out_hook(_mod, _inp, output):
        y = output[0] if isinstance(output, tuple) else output
        if torch.is_tensor(y):
            captured["out"] = y.detach()

    handles = [block.register_forward_hook(out_hook)]
    try:
        def pre_hook_kw(_mod, args, kwargs):
            captured["args"] = tuple(a.detach() if torch.is_tensor(a) else a for a in args)
            captured["kwargs"] = {
                k: (v.detach() if torch.is_tensor(v) else v) for k, v in kwargs.items()
            }
        handles.append(block.register_forward_pre_hook(pre_hook_kw, with_kwargs=True))
    except TypeError:
        def pre_hook(_mod, args):
            captured["args"] = tuple(a.detach() if torch.is_tensor(a) else a for a in args)
            captured["kwargs"] = {}
        handles.append(block.register_forward_pre_hook(pre_hook))
    try:
        with torch.inference_mode():
            try:
                model(**inputs, use_cache=False)
            except TypeError:
                model(**inputs)
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
    kwargs = dict(captured.get("kwargs") or {})
    # remove caches mutáveis (crescem a cada chamada repetida do bloco)
    for k in list(kwargs.keys()):
        if k in _BLOCK_CACHE_KWARGS:
            kwargs.pop(k)
    if "use_cache" in kwargs:
        kwargs["use_cache"] = False
    captured["kwargs"] = kwargs
    # forward direto do bloco funciona com estas entradas?
    direct_ok = False
    if captured.get("args"):
        try:
            with torch.inference_mode():
                out = block(*captured["args"], **kwargs)
            y = out[0] if isinstance(out, tuple) else out
            direct_ok = torch.is_tensor(y)
        except Exception as exc:
            print(f"[C3] AVISO forward direto do bloco indisponível ({exc}); usando forward do modelo")
    captured["direct_ok"] = direct_ok
    return captured


def make_block_forward(model: nn.Module, block: nn.Module, cap: Dict[str, Any],
                       inputs: Dict[str, "torch.Tensor"]):
    """Retorna (fn, modo): fn() executa o bloco sobre os hidden states REAIS e devolve y."""
    if cap.get("direct_ok"):
        args = cap["args"]
        kwargs = cap["kwargs"]

        def fn_direct():
            with torch.inference_mode():
                out = block(*args, **kwargs)
            return out[0] if isinstance(out, tuple) else out

        return fn_direct, "block_direct_real_hidden_states"

    def fn_model():
        holder: Dict[str, "torch.Tensor"] = {}

        def hook(_mod, _inp, output):
            y = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(y):
                holder["y"] = y.detach()

        h = block.register_forward_hook(hook)
        try:
            with torch.inference_mode():
                try:
                    model(**inputs, use_cache=False)
                except TypeError:
                    model(**inputs)
        finally:
            h.remove()
        return holder.get("y")

    return fn_model, "model_forward_block_output_hook"


def make_model_forward(model: nn.Module, watch_block: nn.Module, inputs: Dict[str, "torch.Tensor"]):
    """fn() = forward completo do modelo; devolve a saída do bloco observado."""
    def fn():
        holder: Dict[str, "torch.Tensor"] = {}

        def hook(_mod, _inp, output):
            y = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(y):
                holder["y"] = y.detach()

        h = watch_block.register_forward_hook(hook)
        try:
            with torch.inference_mode():
                try:
                    model(**inputs, use_cache=False)
                except TypeError:
                    model(**inputs)
        finally:
            h.remove()
        return holder.get("y")

    return fn


def forward_logits(model: nn.Module, inputs: Dict[str, "torch.Tensor"]) -> Optional["torch.Tensor"]:
    try:
        with torch.inference_mode():
            out = model(**inputs)
        logits = getattr(out, "logits", None)
        if logits is None and isinstance(out, tuple) and out and torch.is_tensor(out[0]):
            logits = out[0]
        return logits.detach().float().cpu() if torch.is_tensor(logits) else None
    except Exception as exc:
        print(f"[C3] AVISO forward de logits: {exc}")
        return None


def measure_generate(model, tokenizer, prompt: str, device: "torch.device", *,
                     max_new_tokens: int, warmup: int = 2, timed: int = 3) -> Dict[str, Any]:
    """Tok/s REAL de model.generate (greedy) — mesmo protocolo p/ baseline e candidato."""
    enc = tokenizer(prompt, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    gen_kwargs: Dict[str, Any] = dict(max_new_tokens=int(max_new_tokens), do_sample=False)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is not None:
        gen_kwargs["pad_token_id"] = pad_id
    with torch.inference_mode():
        for _ in range(max(0, int(warmup))):
            model.generate(**enc, **{**gen_kwargs, "max_new_tokens": min(8, int(max_new_tokens))})
    if device.type == "cuda":
        torch.cuda.synchronize()
    tok_s_runs: List[float] = []
    last_out = None
    n_new = 0
    for _ in range(max(1, int(timed))):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        with torch.inference_mode():
            out = model.generate(**enc, **gen_kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt_s = (time.perf_counter_ns() - t0) / 1e9
        n_new = int(out.shape[1] - enc["input_ids"].shape[1])
        tok_s_runs.append(n_new / max(dt_s, 1e-9))
        last_out = out
    runs = sorted(tok_s_runs)
    new_ids: List[int] = []
    if last_out is not None:
        new_ids = [int(t) for t in last_out[0][enc["input_ids"].shape[1]:].tolist()]
    return {
        "tok_s_median": runs[len(runs) // 2],
        "tok_s_runs": tok_s_runs,
        "n_new_tokens": n_new,
        "warmup_runs": int(warmup),
        "timed_runs": len(tok_s_runs),
        "greedy": True,
        "max_new_tokens": int(max_new_tokens),
        "new_token_ids": new_ids,
        "method": "model_generate_perf_counter_ns_median_v1",
    }


def token_exact_match(a: List[int], b: List[int]) -> Dict[str, Any]:
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


# ---------------------------------------------------------------------------
# Recorder — grava JSON local (upsert por battery_id) + publica incremental
# ---------------------------------------------------------------------------

def publish_record(rec: Dict[str, Any], endpoint: Optional[str] = None) -> None:
    """Publisher endurecido: HTTPS obrigatório + token >= 32 chars (contrato §5)."""
    endpoint = endpoint or os.environ.get("RIFT_RESULTS_ENDPOINT") or "https://rift-lm.vercel.app/api/results"
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
            "User-Agent": "c3-methodology-battery/1.0",
        })
        with urlopen(req, timeout=60) as resp:
            print(f"[publish] HTTP {resp.status} battery={rec.get('battery_id')}")
    except Exception as exc:
        print(f"[publish] AVISO: {exc}")


class C3Recorder:
    """Passo 13: cada registro alimenta o dashboard assim que é gravado."""

    def __init__(self, out_dir: Path, *, tech: str, tech_upper: str, model_id: str,
                 run_id: str, schema_fields: Dict[str, Any], publish_on: bool,
                 endpoint: Optional[str] = None):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.batteries_dir = out_dir / "batteries"
        self.batteries_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = out_dir / f"c3_{tech}_test_batteries.json"
        self.tech_upper = tech_upper
        self.model_id = model_id
        self.run_id = run_id
        self.schema_fields = schema_fields
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
        self.summary_rows: List[Dict[str, Any]] = []

    def emit(
        self,
        battery_id: str,
        status: str,
        *,
        scope: str,
        quality_output: Optional[Dict[str, Any]] = None,
        full_gate: Optional[bool] = None,
        metrics: Optional[Dict[str, Any]] = None,
        notes: str = "",
        primary: bool = False,
        demote: bool = False,
        ram_base: Optional[Dict[str, Any]] = None,
        ram_cand: Optional[Dict[str, Any]] = None,
        baseline_disk: Optional[int] = None,
        candidate_disk: Optional[int] = None,
        baseline_tok_s: Optional[float] = None,
        candidate_tok_s: Optional[float] = None,
        implementation: Optional[Dict[str, Any]] = None,
        highlight: str = "",
    ) -> Dict[str, Any]:
        is_primary = bool(primary and not demote)
        if full_gate is None:
            full_gate = status in ("PASS", "EXPERIMENTAL_PASS")
        rec = {
            "timestamp_utc": utc(),
            "run_id": self.run_id,
            "technology": self.tech_upper,
            "model_id": self.model_id,
            "battery_id": battery_id,
            "status": status,
            **self.schema_fields,
            "comparison_role": "primary" if is_primary else None,
            "eligible_for_primary_ranking": is_primary,
            "baseline_ram_bytes": ram_top_level(ram_base),
            "candidate_ram_bytes": ram_top_level(ram_cand),
            "baseline_disk_bytes": baseline_disk,
            "candidate_disk_bytes": candidate_disk,
            "baseline_tok_s": baseline_tok_s,
            "candidate_tok_s": candidate_tok_s,
            "measurement_scope": scope,
            "quality": {"full_local_gate_pass": bool(full_gate), "output": quality_output},
            "metrics": metrics or {},
            "notes": notes[:1200],
        }
        if implementation is not None:
            rec["implementation"] = implementation
        path = self.batteries_dir / f"{self.run_id}__{battery_id}.json"
        path.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        # upsert pelo par (model_id, battery_id) no JSON agregado — rodar outro
        # --model no mesmo out dir não pode apagar o histórico do modelo anterior
        # (a seleção do WINNER é chaveada por par modelo|battery_id, contrato §1)
        self.records = [
            r for r in self.records
            if r.get("battery_id") != battery_id or r.get("model_id") != self.model_id
        ]
        self.records.append(rec)
        self.records.sort(key=lambda r: str(r.get("battery_id")))
        self.json_path.write_text(
            json.dumps(self.records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"[BATTERY] {battery_id} {status} -> {path}")
        if self.publish_on:
            publish_record(rec, self.endpoint)
        self.summary_rows.append({"battery_id": battery_id, "status": status, "highlight": highlight})
        return rec

# ---------------------------------------------------------------------------
# Orquestração dos 16 passos
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="C3 — metodologia C1 de 16 passos (contrato C3_CONTRACTS_V1 §2)")
    p.add_argument("--technology", required=True, choices=sorted(TECH_DEFAULT_CODEC))
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--codec", default=None, choices=sorted(_CODECS))
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument("--gate-percentile", type=float, default=70.0)
    p.add_argument("--blocks", type=int, default=4, help="nº de blocos consecutivos do passo 15")
    p.add_argument("--max-new-tokens", type=int, default=64, help="tokens do generate do passo 16")
    p.add_argument("--out", default="c3_test_output")
    p.add_argument("--publish", default="on", choices=["on", "off"])
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--skip-full-model", action="store_true",
                   help="pula o passo 16 (generate e2e) em ambientes com pouca RAM")
    p.add_argument("--results-endpoint", default=None, help="URL HTTPS /api/results (default: env)")
    values = sys.argv[1:] if argv is None else list(argv)
    args = p.parse_args(without_ipykernel_connection_args(values))
    if not 0 <= args.gate_percentile <= 100:
        p.error("--gate-percentile precisa estar entre 0 e 100")
    if args.trust_remote_code:
        print("[C3] AVISO: --trust-remote-code executa código do repositório do modelo. "
              "Use apenas com modelos de fonte confiável.")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    bootstrap_colab_secrets()

    tech = args.technology.lower()
    tech_upper = tech.upper()
    codec_name = (args.codec or TECH_DEFAULT_CODEC[tech]).lower()
    codec = get_codec(codec_name)
    device = resolve_device(args.device)
    token = ensure_hf_login()
    model_id = args.model.strip().replace("https://huggingface.co/", "").strip("/")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = out_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id()
    schema_fields = schema_v2_fields(model_id, device, codec_name)
    publish_on = args.publish != "off"
    recorder = C3Recorder(
        out_dir, tech=tech, tech_upper=tech_upper, model_id=model_id, run_id=run_id,
        schema_fields=schema_fields, publish_on=publish_on, endpoint=args.results_endpoint,
    )

    def bid(suffix: str) -> str:
        return f"C3_{tech_upper}_{suffix}"

    print(f"[C3] tech={tech_upper} codec={codec_name} model={model_id} device={device} "
          f"rank={args.rank} gs={args.group_size} gate_pct={args.gate_percentile}")

    # resumo p/ decisão (passo 14) e gain report
    summary: Dict[str, Any] = {
        "run_id": run_id, "technology": tech_upper, "codec": codec_name,
        "model_id": model_id, "device": device.type,
        "benchmark_protocol": BENCHMARK_PROTOCOL,
    }

    # ------------------------------------------------------------------
    # Carrega o modelo real
    # ------------------------------------------------------------------
    try:
        model, tokenizer = load_model(model_id, device, args.trust_remote_code, token)
    except Exception as exc:
        traceback.print_exc()
        recorder.emit(
            bid("LOAD_MODEL"), "FAIL",
            scope="C3 model load (diagnóstico de infraestrutura)",
            metrics={"error": str(exc)[:800]},
            notes=f"Falha ao carregar {model_id}: {exc}",
            highlight="load falhou",
        )
        return 0

    layer_name, weight = find_linear_weight(model)
    weight = weight.to(dtype=torch.float32).cpu()
    print(f"[C3] Linear alvo: {layer_name} shape={tuple(weight.shape)}")

    # ativação REAL via forward hook (prompt fixo PT-BR); fallback sintético rebaixa
    x = capture_activation(model, tokenizer, layer_name, device, ACTIVATION_PROMPT)
    activation_source = "real_model_activation"
    if x is None or x.ndim != 2 or x.shape[-1] != weight.shape[1]:
        print("[C3] AVISO: sem ativação real — fallback sintético (registros Linear rebaixados)")
        x = torch.randn(32, weight.shape[1], dtype=torch.float32)
        activation_source = "synthetic_fallback"
    elif x.shape[0] > 64:
        x = x[:64].contiguous()
    x = x.to(dtype=torch.float32).cpu()
    demote_linear = activation_source != "real_model_activation"
    summary["activation_source"] = activation_source

    # ------------------------------------------------------------------
    # Decomposição da Linear alvo com o codec da tecnologia
    # ------------------------------------------------------------------
    print(f"[C3] decompondo F0 ({codec_name}) + F1 low-rank...")
    stages, packed = decompose_with_codec(weight, codec, rank=args.rank, group_size=args.group_size)
    print(f"[C3] F0={stages.f0_bytes} B  F1={stages.f1_bytes} B  baseline={stages.baseline_bytes} B  "
          f"reduction={stages.to_meta()['disk_reduction_pct']:.1f}%")

    # ------------------------------------------------------------------
    # Passos 1, 5 — Bundle M0 congelado + golden tests
    # ------------------------------------------------------------------
    bundle_path = out_dir / "model.cascade"
    golden: Dict[str, Any] = {"golden_pass": False}
    bundle_status = "FAIL"
    try:
        bundle_meta = write_cascade_bundle(
            bundle_path, stages=stages, model_id=model_id,
            target_layer=layer_name, gate_percentile=args.gate_percentile,
            f0_codec=F0_BUNDLE_CODEC_LABEL[codec_name],
        )
        bundle_buf = bundle_path.read_bytes()
        golden = run_bundle_golden_tests(bundle_buf)
        bundle_status = "PASS" if golden.get("golden_pass") else "FAIL"
        bundle_bytes = stat_bytes(bundle_path)
        recorder.emit(
            bid("BUNDLE_M0_FREEZE"), bundle_status,
            scope=(
                "Bundle CSCD v0x0003 (header 128B) gravado da Linear real decomposta; "
                "golden tests: válido PASS + negativos (magic, versão, header truncado, "
                "CRC, stage offset fora dos limites) rejeitados pelo leitor validador"
            ),
            quality_output={"golden_tests": {k: v for k, v in golden.items() if k != "golden_pass"}},
            full_gate=golden.get("golden_pass"),
            metrics={
                "bundle": {
                    "path": str(bundle_path), "file_size": bundle_meta["file_size"],
                    "checksum": bundle_meta["checksum"], "n_stages": len(bundle_meta["stages"]),
                    "codec_f0": codec_name,
                },
                "golden": golden,
            },
            baseline_disk=stages.baseline_bytes,
            candidate_disk=bundle_bytes,
            notes=(
                f"Passos 1+5: bundle real ({bundle_bytes} B, os.stat) da Linear {layer_name}; "
                f"F0 codec={codec_name}. Golden: 1 caso válido + 5 negativos, todos verificados."
            ),
            highlight=f"golden={'OK' if golden.get('golden_pass') else 'FALHOU'}",
        )
    except Exception as exc:
        traceback.print_exc()
        bundle_buf = b""
        recorder.emit(
            bid("BUNDLE_M0_FREEZE"), "FAIL",
            scope="Bundle M0 freeze (falha de escrita/validação)",
            metrics={"error": str(exc)[:800]},
            notes=f"Falha no bundle: {exc}",
            highlight="erro",
        )
    summary["bundle_golden_pass"] = bundle_status == "PASS"

    # ------------------------------------------------------------------
    # Passo 2 — Stage Table/Page ABI congelada (entrada de 24 bytes)
    # ------------------------------------------------------------------
    stage_status = "FAIL"
    try:
        if not bundle_buf:
            raise RuntimeError("bundle indisponível")
        stage_checks = run_stage_page_checks(
            bundle_buf, expected_f0_codec=F0_BUNDLE_CODEC_LABEL[codec_name]
        )
        stage_status = "PASS" if stage_checks.get("pass") else "FAIL"
        recorder.emit(
            bid("STAGE_PAGE_M0_FREEZE"), stage_status,
            scope=(
                "Stage Table ABI congelada: entradas de 24B (<QQII) parseadas do bundle real; "
                "contagem/ordem/alinhamento-64B/limites/tipos/codec-F0 declarado "
                "+ negativo de size corrompido"
            ),
            quality_output={"checks": stage_checks},
            full_gate=stage_checks.get("pass"),
            metrics={"stage_page": stage_checks},
            baseline_disk=stages.baseline_bytes,
            candidate_disk=stat_bytes(bundle_path) if bundle_path.is_file() else None,
            notes=(
                "Passo 2: ABI da stage table verificada sobre o arquivo real "
                "(2 entradas: BASE_STAGE + RESIDUAL_LOWRANK) e rejeição do negativo."
            ),
            highlight=f"ABI={'OK' if stage_status == 'PASS' else 'FALHOU'}",
        )
    except Exception as exc:
        traceback.print_exc()
        recorder.emit(
            bid("STAGE_PAGE_M0_FREEZE"), "FAIL",
            scope="Stage Table ABI (falha)",
            metrics={"error": str(exc)[:800]},
            notes=f"Falha no parse da stage table: {exc}",
            highlight="erro",
        )
    summary["stage_page_pass"] = stage_status == "PASS"

    # ------------------------------------------------------------------
    # Passo 4 — CASCADE-IR v3 write→validate→reload
    # ------------------------------------------------------------------
    ir_status = "FAIL"
    try:
        ir_info = run_ir_roundtrip(out_dir, model_id, layer_name)
        ir_status = "PASS" if ir_info.get("pass") else "FAIL"
        recorder.emit(
            bid("IR_WRITER"), ir_status,
            scope="CASCADE-IR v3: make_linear_ir → validate → write JSON → reload → validate → igualdade",
            quality_output={"roundtrip_equal": ir_info["roundtrip_equal"]},
            full_gate=ir_info.get("pass"),
            metrics={"ir": ir_info},
            notes=f"Passo 4: roundtrip do IR da op Linear real ({layer_name}); ir_version={IR_VERSION}.",
            highlight=f"roundtrip={'OK' if ir_status == 'PASS' else 'FALHOU'}",
        )
    except Exception as exc:
        traceback.print_exc()
        recorder.emit(
            bid("IR_WRITER"), "FAIL",
            scope="CASCADE-IR v3 roundtrip (falha)",
            metrics={"error": str(exc)[:800]},
            notes=f"Falha no roundtrip do IR: {exc}",
            highlight="erro",
        )
    summary["ir_writer_pass"] = ir_status == "PASS"

    # ------------------------------------------------------------------
    # Passo 6 — Leitor C++ mmap sobre o bundle real (POSIX; SKIPPED no Windows)
    # ------------------------------------------------------------------
    try:
        cpp = run_cpp_bundle_reader(bundle_path, out_dir)
    except Exception as exc:
        cpp = {"status": "FAIL", "ran": False, "note": f"erro inesperado: {exc}"}
    cpp_impl = dict(schema_fields["implementation"])
    if cpp.get("ran"):
        cpp_impl = {"kind": "NATIVE_MEASURED", "native": True, "simulated": False}
    recorder.emit(
        bid("CPP_BUNDLE_READER"), cpp["status"],
        scope=(
            "Leitor C++ mmap (cascade/runtime/cpp/mmap_smoke.cpp, g++ -std=c++20 -O2) "
            "executado sobre o bundle real; wall-clock de read+validate; "
            "POSIX-only — SKIPPED no Windows/sem compilador"
        ),
        quality_output={"exit_code": cpp.get("exit_code"), "ran": cpp.get("ran", False)},
        full_gate=cpp["status"] == "PASS",
        metrics={"cpp_reader": cpp},
        implementation=cpp_impl,
        notes=("Passo 6: " + str(cpp.get("note", "")))[:1200],
        highlight=(
            f"{cpp.get('read_validate_ms', 0):.1f} ms" if cpp.get("ran") else "SKIPPED"
        ),
    )
    summary["cpp_reader"] = {"status": cpp["status"], "ran": bool(cpp.get("ran"))}

    # ------------------------------------------------------------------
    # Passos 3, 7-12 — Linear real, 4 caminhos (um registro por caminho)
    # ------------------------------------------------------------------
    lin_dir = artifacts_dir / "linear"
    lin_dir.mkdir(parents=True, exist_ok=True)
    w_orig_path = lin_dir / "linear_w_original.bin"
    w_orig_path.write_bytes(tensor_bytes(weight))
    lin_baseline_disk = stat_bytes(w_orig_path)  # artefato binário REAL (os.stat)
    lin_art = write_stage_artifacts(lin_dir, "linear_target", stages)

    runtime = C3LinearRuntime(stages, packed, codec, gate_percentile=args.gate_percentile)
    x_cpu = x
    w_cpu = weight
    cpu_dev = torch.device("cpu")

    with torch.inference_mode():
        y_ref = F.linear(x_cpu, w_cpu)
        r_f0 = runtime.execute(x_cpu, path="F0_ONLY")
        r_full = runtime.execute(x_cpu, path="F0_PLUS_F1_ALWAYS")
        r_gate = runtime.execute(x_cpu, path="F0_GATE_F1")

    q_f0 = cosine_nrmse(y_ref, r_f0["y"])
    q_full = cosine_nrmse(y_ref, r_full["y"])
    q_gate = cosine_nrmse(y_ref, r_gate["y"])

    print("[C3] benchmark Linear 4 caminhos (warmup>=5, iters>=30, perf_counter_ns)...")
    perf_orig, ram_lin_orig = measure_phase_ram(
        lambda: benchmark_ns(lambda: F.linear(x_cpu, w_cpu), warmup=5, iterations=30, device=cpu_dev))
    perf_f0, ram_lin_f0 = measure_phase_ram(
        lambda: benchmark_ns(lambda: runtime.execute(x_cpu, path="F0_ONLY"), warmup=5, iterations=30, device=cpu_dev))
    perf_full, ram_lin_full = measure_phase_ram(
        lambda: benchmark_ns(lambda: runtime.execute(x_cpu, path="F0_PLUS_F1_ALWAYS"), warmup=5, iterations=30, device=cpu_dev))
    perf_gate, ram_lin_gate = measure_phase_ram(
        lambda: benchmark_ns(lambda: runtime.execute(x_cpu, path="F0_GATE_F1"), warmup=5, iterations=30, device=cpu_dev))

    io_bytes = int((x_cpu.numel() + y_ref.numel()) * 4)
    est_lin_base = stages.baseline_bytes + io_bytes
    gate_rate = 1.0 - r_gate["f1_skip_rate"]

    def emit_linear(suffix: str, status: str, *, path_name: str, q: Dict[str, float],
                    perf: Dict[str, Any], ram_phase, counters: Dict[str, Any],
                    cand_disk: Optional[int], est_cand: Optional[int],
                    extra: Optional[Dict[str, Any]] = None, primary: bool = False,
                    note: str = "") -> None:
        metrics = {
            "operation": {
                "metric": "linear_latency",
                "baseline_median_ms_proxy": perf_orig["median_ms"],
                "candidate_median_ms_proxy": perf["median_ms"],
                "speedup_x_proxy": perf_orig["median_ms"] / max(perf["median_ms"], 1e-12),
                "latency": perf,
                "rows_processed": int(x_cpu.shape[0]),
                "path": path_name,
                "bench_device": "cpu",
                "tok_s_note": "proxy de Linear; tok/s de topo só no FULLMODEL_E2E_TOKS",
            },
            "memory": memory_metrics(
                ram_lin_orig, ram_phase,
                estimated_baseline=est_lin_base, estimated_candidate=est_cand,
            ),
            "cascade": {
                **stages.to_meta(),
                "codec_f0": codec_name,
                "codec_meta": packed.meta,
                "target_layer": layer_name,
                "activation_source": activation_source,
                "activation_rows": int(x_cpu.shape[0]),
                "artifacts": {"w_original_bytes": lin_baseline_disk, **lin_art},
                **counters,
                **(extra or {}),
            },
        }
        if tech == "spectra":
            metrics["drift_contract"] = {
                "metric": "nrmse_vs_original", "value": q["nrmse"],
                "limit": 0.05, "pass": q["nrmse"] <= 0.05,
            }
        recorder.emit(
            bid(suffix), status,
            scope=(
                f"C3 Linear real 4-caminhos path={path_name}; layer={layer_name}; "
                f"F0 codec={codec_name} + F1 low-rank; benchmark CPU perf_counter_ns "
                f"(warmup=5, iters=30); RAM topo=null (registro de operação — adendo V3: "
                f"RSS de topo só no FULLMODEL_E2E_TOKS); fases VmRSS em metrics.memory"
            ),
            quality_output=q,
            full_gate=status in ("PASS", "EXPERIMENTAL_PASS"),
            metrics=metrics,
            baseline_disk=lin_baseline_disk,
            candidate_disk=cand_disk,
            primary=primary,
            demote=demote_linear,
            notes=(note + (" | activation_source=synthetic_fallback (registro rebaixado)"
                           if demote_linear else ""))[:1200],
            highlight=f"cos={q['cosine']:.4f} {perf['median_ms']:.3f}ms",
        )

    emit_linear(
        "LINEAR_ORIGINAL", "PASS", path_name="ORIGINAL",
        q={"cosine": 1.0, "nrmse": 0.0},
        perf=perf_orig, ram_phase=ram_lin_orig,
        counters={"F0_calls": 0, "F1_calls": 0, "F1_skip_rate": None},
        cand_disk=None, est_cand=est_lin_base,
        note="Caminho A: F.linear com W denso original (referência de qualidade/latência).",
    )
    emit_linear(
        "LINEAR_F0_ONLY",
        "EXPERIMENTAL_PASS" if q_f0["cosine"] >= 0.90 else "EXPERIMENTAL_FAIL",
        path_name="F0_ONLY", q=q_f0, perf=perf_f0, ram_phase=ram_lin_f0,
        counters={"F0_calls": r_f0["f0_calls"], "F1_calls": 0, "F1_skip_rate": 1.0},
        cand_disk=lin_art["f0_bytes"], est_cand=stages.f0_bytes + io_bytes,
        note=f"Caminho B: somente F0 {codec_name}. Ganho bruto de quantização.",
    )
    emit_linear(
        "LINEAR_F0_PLUS_F1_ALWAYS",
        "EXPERIMENTAL_PASS" if q_full["cosine"] >= 0.98 else "EXPERIMENTAL_FAIL",
        path_name="F0_PLUS_F1_ALWAYS", q=q_full, perf=perf_full, ram_phase=ram_lin_full,
        counters={"F0_calls": r_full["f0_calls"], "F1_calls": r_full["f1_calls"], "F1_skip_rate": 0.0},
        cand_disk=lin_art["total_bytes"], est_cand=stages.f0_bytes + stages.f1_bytes + io_bytes,
        note="Caminho C: F0+F1 always. Qualidade recuperada pelo residual low-rank.",
    )
    lin_gate_pass = (
        q_gate["cosine"] >= 0.995 and q_gate["nrmse"] <= 0.05 and r_gate["f1_skip_rate"] > 0
    )
    emit_linear(
        "LINEAR_F0_GATE_F1",
        "PASS" if lin_gate_pass else "EXPERIMENTAL_FAIL",
        path_name="F0_GATE_F1", q=q_gate, perf=perf_gate, ram_phase=ram_lin_gate,
        counters={
            "F0_calls": r_gate["f0_calls"], "F1_calls": r_gate["f1_calls"],
            "F1_skip_rate": r_gate["f1_skip_rate"],
        },
        cand_disk=lin_art["total_bytes"],
        est_cand=stages.f0_bytes + int(round(gate_rate * stages.f1_bytes)) + io_bytes,
        extra={"gate": r_gate.get("gate")},
        primary=True,
        note=(
            f"Caminho D: F0+Gate·F1. skip={r_gate['f1_skip_rate']:.3f} "
            f"gate_rate={gate_rate:.3f} (critério C1: cos>=0.995, nrmse<=0.05, skip>0)."
        ),
    )
    summary["linear_gated"] = {
        "cosine": q_gate["cosine"], "nrmse": q_gate["nrmse"],
        "f1_skip_rate": r_gate["f1_skip_rate"],
        "median_ms": perf_gate["median_ms"], "baseline_median_ms": perf_orig["median_ms"],
        "pass": lin_gate_pass, "demoted": demote_linear,
    }

    # ------------------------------------------------------------------
    # Passos 11-12 — Bloco Transformer real, 4 caminhos
    # ------------------------------------------------------------------
    decomp_cache: Dict[str, Tuple[CascadeLinearStages, PackedLinear]] = {}
    artifact_registry: Dict[str, Dict[str, Any]] = {}
    artifacts_model_dir = artifacts_dir / "model"

    inputs_small = tokenizer(ACTIVATION_PROMPT, return_tensors="pt", truncation=True, max_length=64)
    inputs_small = {k: v.to(device) for k, v in inputs_small.items()}
    n_tokens_small = int(inputs_small["input_ids"].numel())
    patch_device = device if getattr(model, "hf_device_map", None) is None else None

    blocks = find_transformer_blocks(model)
    summary["block_gated"] = {"cosine": 0.0, "nrmse": 1.0, "f1_skip_rate": 0.0, "pass": False}

    if not blocks:
        for sfx in ("BLOCK_ORIGINAL", "BLOCK_F0_ONLY", "BLOCK_F0_PLUS_F1_ALWAYS", "BLOCK_F0_GATE_F1"):
            recorder.emit(
                bid(sfx), "FAIL",
                scope="C3 bloco Transformer real (indisponível)",
                metrics={"error": "nenhum transformer block encontrado no modelo"},
                notes="find_transformer_blocks não localizou blocos — caminho de bloco FAIL.",
                highlight="sem blocos",
            )
    else:
        block_name, block = blocks[0]
        print(f"[C3] bloco alvo [0] {block_name}")
        cap = capture_block_io(model, block, inputs_small)
        block_fn, block_mode = make_block_forward(model, block, cap, inputs_small)
        y_base = block_fn()
        if y_base is None and torch.is_tensor(cap.get("out")):
            y_base = cap["out"]
        if y_base is None:
            for sfx in ("BLOCK_ORIGINAL", "BLOCK_F0_ONLY", "BLOCK_F0_PLUS_F1_ALWAYS", "BLOCK_F0_GATE_F1"):
                recorder.emit(
                    bid(sfx), "FAIL",
                    scope="C3 bloco Transformer real (forward sem saída)",
                    metrics={"error": "forward do bloco não produziu saída"},
                    notes="Nem forward direto nem hook de saída capturaram o hidden state do bloco.",
                    highlight="sem saída",
                )
        else:
            block_iters, block_warm = (12, 3) if block_mode == "block_direct_real_hidden_states" else (6, 2)
            print(f"[C3] baseline do bloco original ({block_mode})...")
            perf_b_orig, ram_b_orig = measure_phase_ram(
                lambda: benchmark_ns(block_fn, warmup=block_warm, iterations=block_iters, device=device))

            print(f"[C3] patch das Linears do bloco (codec={codec_name}, W fora do caminho quente)...")
            path_results: Dict[str, Dict[str, Any]] = {}
            block_error: Optional[str] = None
            replaced: Dict[str, nn.Module] = {}
            block_base_disk = block_cand_disk = block_f0_disk = None
            resident_with_cache = resident_no_cache = 0
            # espelha os demais passos: qualquer exceção aqui vira FAIL registrado
            # (nunca propaga para fora de main sem registros BLOCK_*/C1_DECISION)
            try:
                originals, replaced = patch_one_block(
                    block, block_name, codec,
                    rank=args.rank, group_size=args.group_size,
                    gate_percentile=args.gate_percentile, path="F0_ONLY",
                    device=patch_device, decomp_cache=decomp_cache,
                    artifacts_dir=artifacts_model_dir, artifact_registry=artifact_registry,
                )
                try:
                    block_base_disk = sum(int(decomp_cache[f][0].baseline_bytes) for f in replaced)
                    block_cand_disk = sum(int(artifact_registry[f]["total_bytes"]) for f in replaced)
                    block_f0_disk = sum(int(artifact_registry[f]["f0_bytes"]) for f in replaced)
                    for path_name in ("F0_ONLY", "F0_PLUS_F1_ALWAYS", "F0_GATE_F1"):
                        set_replaced_path(replaced, path_name)
                        # contadores zerados ANTES e lidos DEPOIS de CADA caminho
                        # (corrige a classe de bug de fechamento do C1)
                        reset_replaced_counters(replaced)
                        y_c = block_fn()
                        perf_c, ram_c = measure_phase_ram(
                            lambda: benchmark_ns(block_fn, warmup=block_warm, iterations=block_iters, device=device))
                        counters = read_replaced_counters(replaced)
                        qy = cosine_nrmse(y_base, y_c) if y_c is not None else {"cosine": 0.0, "nrmse": 1.0}
                        gate_rates = [float(getattr(m, "last_gate_rate", 0.0)) for m in replaced.values()]
                        path_results[path_name] = {
                            "q": qy, "perf": perf_c, "ram": ram_c, "counters": counters,
                            "gate_rate_mean": sum(gate_rates) / max(len(gate_rates), 1),
                        }
                    resident_with_cache = sum(int(m.stats()["resident_bytes_with_cache"]) for m in replaced.values())
                    resident_no_cache = sum(int(m.stats()["resident_bytes"]) for m in replaced.values())
                finally:
                    restore_block_linears(block, originals, block_name)
            except Exception as exc:
                traceback.print_exc()
                block_error = f"{type(exc).__name__}: {exc}"

            if block_error is not None:
                for sfx in ("BLOCK_ORIGINAL", "BLOCK_F0_ONLY", "BLOCK_F0_PLUS_F1_ALWAYS", "BLOCK_F0_GATE_F1"):
                    recorder.emit(
                        bid(sfx), "FAIL",
                        scope="C3 bloco Transformer real (erro em patch/execução)",
                        metrics={"error": block_error[:800]},
                        notes=(
                            f"Falha nos passos 11-12: {block_error}. Modelo restaurado; "
                            "block_gated mantém o default reprovado para a decisão C1."
                        )[:1200],
                        highlight="erro",
                    )
            else:
                def emit_block(suffix: str, status: str, *, path_name: str, q: Dict[str, float],
                               perf: Dict[str, Any], ram_phase, counters: Dict[str, Any],
                               cand_disk: Optional[int], extra: Optional[Dict[str, Any]] = None,
                               primary: bool = False, note: str = "") -> None:
                    metrics = {
                        "operation": {
                            "metric": "block_forward_latency",
                            "baseline_median_ms_proxy": perf_b_orig["median_ms"],
                            "candidate_median_ms_proxy": perf["median_ms"],
                            "speedup_x_proxy": perf_b_orig["median_ms"] / max(perf["median_ms"], 1e-12),
                            "latency": perf,
                            "forward_mode": block_mode,
                            "n_tokens": n_tokens_small,
                            "path": path_name,
                            "tok_s_note": "proxy de bloco; tok/s de topo só no FULLMODEL_E2E_TOKS",
                        },
                        "memory": memory_metrics(ram_b_orig, ram_phase),
                        "cascade": {
                            "block_name": block_name,
                            "block_index": 0,
                            "n_linears_patched": len(replaced),
                            "codec_f0": codec_name,
                            "resident_stage_bytes_with_cache": resident_with_cache,
                            "resident_stage_bytes_no_cache": resident_no_cache,
                            "original_weight_on_hot_path": False,
                            **counters,
                            **(extra or {}),
                        },
                    }
                    if tech == "spectra":
                        metrics["drift_contract"] = {
                            "metric": "nrmse_vs_original_block", "value": q["nrmse"],
                            "limit": 0.05, "pass": q["nrmse"] <= 0.05,
                        }
                    recorder.emit(
                        bid(suffix), status,
                        scope=(
                            f"C3 bloco Transformer real 4-caminhos path={path_name}; "
                            f"block={block_name}; hidden states REAIS ({block_mode}); "
                            f"F0 codec={codec_name}+Gate·F1; RAM topo=null (registro de "
                            f"operação — adendo V3: RSS de topo só no FULLMODEL_E2E_TOKS); "
                            f"fases VmRSS em metrics.memory"
                        ),
                        quality_output=q,
                        full_gate=status in ("PASS", "EXPERIMENTAL_PASS"),
                        metrics=metrics,
                        baseline_disk=block_base_disk,
                        candidate_disk=cand_disk,
                        primary=primary,
                        notes=note,
                        highlight=f"cos={q['cosine']:.4f} {perf['median_ms']:.2f}ms",
                    )

                emit_block(
                    "BLOCK_ORIGINAL", "PASS", path_name="ORIGINAL",
                    q={"cosine": 1.0, "nrmse": 0.0}, perf=perf_b_orig, ram_phase=ram_b_orig,
                    counters={"F0_calls": 0, "F1_calls": 0, "F1_skip_rate": None},
                    cand_disk=None,
                    note="Caminho A do bloco: forward com as Linears densas originais (referência).",
                )
                r_bf0 = path_results["F0_ONLY"]
                emit_block(
                    "BLOCK_F0_ONLY",
                    "EXPERIMENTAL_PASS" if r_bf0["q"]["cosine"] >= 0.80 else "EXPERIMENTAL_FAIL",
                    path_name="F0_ONLY", q=r_bf0["q"], perf=r_bf0["perf"], ram_phase=r_bf0["ram"],
                    counters=r_bf0["counters"], cand_disk=block_f0_disk,
                    note=f"Caminho B do bloco: somente F0 {codec_name} em todas as Linears.",
                )
                r_bfull = path_results["F0_PLUS_F1_ALWAYS"]
                emit_block(
                    "BLOCK_F0_PLUS_F1_ALWAYS",
                    "EXPERIMENTAL_PASS" if r_bfull["q"]["cosine"] >= 0.95 else "EXPERIMENTAL_FAIL",
                    path_name="F0_PLUS_F1_ALWAYS", q=r_bfull["q"], perf=r_bfull["perf"],
                    ram_phase=r_bfull["ram"], counters=r_bfull["counters"], cand_disk=block_cand_disk,
                    note="Caminho C do bloco: F0+F1 always em todas as Linears.",
                )
                r_bgate = path_results["F0_GATE_F1"]
                # critério C1 do contrato (§2/docs/C3_METHODOLOGY.md): cosine>=0.995
                # E NRMSE<=0.05 no caminho gated também para o BLOCO (não 0.98)
                block_gate_pass = (
                    r_bgate["q"]["cosine"] >= 0.995
                    and r_bgate["q"]["nrmse"] <= 0.05
                    and r_bgate["counters"]["F1_skip_rate"] > 0
                )
                emit_block(
                    "BLOCK_F0_GATE_F1",
                    "PASS" if block_gate_pass else "EXPERIMENTAL_FAIL",
                    path_name="F0_GATE_F1", q=r_bgate["q"], perf=r_bgate["perf"],
                    ram_phase=r_bgate["ram"], counters=r_bgate["counters"], cand_disk=block_cand_disk,
                    extra={"gate_rate_mean": r_bgate["gate_rate_mean"]},
                    primary=True,
                    note=(
                        f"Caminho D do bloco: F0+Gate·F1. cos={r_bgate['q']['cosine']:.4f} "
                        f"nrmse={r_bgate['q']['nrmse']:.4f} "
                        f"skip={r_bgate['counters']['F1_skip_rate']:.3f} "
                        f"(critério C1 bloco: cos>=0.995, nrmse<=0.05, skip>0)."
                    ),
                )
                summary["block_gated"] = {
                    "cosine": r_bgate["q"]["cosine"], "nrmse": r_bgate["q"]["nrmse"],
                    "f1_skip_rate": r_bgate["counters"]["F1_skip_rate"],
                    "median_ms": r_bgate["perf"]["median_ms"],
                    "baseline_median_ms": perf_b_orig["median_ms"],
                    "pass": block_gate_pass,
                }

    # ------------------------------------------------------------------
    # Passo 14 — C3_<TECH>_C1_DECISION (aprova/reprova C1)
    # ------------------------------------------------------------------
    lg = summary["linear_gated"]
    bg = summary["block_gated"]
    golden_ok = bool(summary["bundle_golden_pass"] and summary["stage_page_pass"])
    criteria = {
        "golden_tests_pass": {"pass": golden_ok,
                              "bundle": summary["bundle_golden_pass"],
                              "stage_page": summary["stage_page_pass"]},
        "linear_gated_cosine_ge_0995": {"pass": lg["cosine"] >= 0.995, "value": lg["cosine"], "min": 0.995},
        "linear_gated_nrmse_le_005": {"pass": lg["nrmse"] <= 0.05, "value": lg["nrmse"], "max": 0.05},
        "block_gated_cosine_ge_0995": {"pass": bg["cosine"] >= 0.995, "value": bg["cosine"], "min": 0.995},
        "block_gated_nrmse_le_005": {"pass": bg["nrmse"] <= 0.05, "value": bg["nrmse"], "max": 0.05},
        "f1_skip_rate_gt_zero": {
            "pass": (lg["f1_skip_rate"] or 0) > 0 and (bg["f1_skip_rate"] or 0) > 0,
            "linear": lg["f1_skip_rate"], "block": bg["f1_skip_rate"],
        },
    }
    decision_pass = all(c["pass"] for c in criteria.values())
    recorder.emit(
        bid("C1_DECISION"), "PASS" if decision_pass else "FAIL",
        scope=(
            "Passo 14 — decisão C1: golden tests ∧ Linear gated (cos>=0.995, nrmse<=0.05) "
            "∧ Bloco gated (cos>=0.995, nrmse<=0.05) ∧ F1_skip_rate>0"
        ),
        quality_output={"decision_pass": decision_pass},
        full_gate=decision_pass,
        metrics={"decision": {
            "criteria": criteria,
            "pass": decision_pass,
            "ir_writer_pass_informative": summary["ir_writer_pass"],
            "cpp_reader": summary["cpp_reader"],
            "linear_gated": lg,
            "block_gated": bg,
        }},
        notes=(
            f"C1 {'APROVADO' if decision_pass else 'REPROVADO'}: "
            + "; ".join(f"{k}={'OK' if v['pass'] else 'X'}" for k, v in criteria.items())
        ),
        highlight="APROVADO" if decision_pass else "REPROVADO",
    )
    summary["c1_decision_pass"] = decision_pass

    # ------------------------------------------------------------------
    # Passo 15 — C3_<TECH>_BLOCKS4_GATED (4 blocos consecutivos patchados)
    # ------------------------------------------------------------------
    summary["blocks4"] = {"pass": False}
    if not decision_pass:
        # docs/C3_METHODOLOGY.md: "A decisão C1 reprova a expansão: os passos 15 e 16
        # só executam para tecnologias com C3_<TECH>_C1_DECISION = PASS."
        recorder.emit(
            bid("BLOCKS4_GATED"), "SKIPPED",
            scope="C3 passo 15 (bloqueado pela decisão C1 reprovada)",
            metrics={"skipped": True, "c1_decision_pass": False},
            notes=(
                f"Passo 15 não executado: {bid('C1_DECISION')} = FAIL reprova a expansão "
                "(docs/C3_METHODOLOGY.md). Registro mantido como SKIPPED para completar "
                "os 16 passos sem publicar métricas primárias."
            ),
            highlight="SKIPPED (C1 reprovado)",
        )
        summary["blocks4"]["skipped"] = True
        summary["blocks4"]["skip_reason"] = "c1_decision_fail"
    elif not blocks:
        recorder.emit(
            bid("BLOCKS4_GATED"), "FAIL",
            scope="C3 4 blocos gated (indisponível)",
            metrics={"error": "nenhum transformer block encontrado"},
            notes="Sem blocos para o passo 15.",
            highlight="sem blocos",
        )
    else:
        n4 = max(1, min(int(args.blocks), len(blocks)))
        watch_name, watch_block = blocks[n4 - 1]
        print(f"[C3] passo 15: {n4} blocos consecutivos gated (observando saída de {watch_name})...")
        fn4 = make_model_forward(model, watch_block, inputs_small)
        y4_ref = fn4()
        perf4_base, ram4_base = measure_phase_ram(
            lambda: benchmark_ns(fn4, warmup=2, iterations=7, device=device))
        patched4: List[Tuple[nn.Module, str, Dict[str, nn.Linear], Dict[str, nn.Module]]] = []
        try:
            for i in range(n4):
                bn_i, blk_i = blocks[i]
                orig_i, repl_i = patch_one_block(
                    blk_i, bn_i, codec,
                    rank=args.rank, group_size=args.group_size,
                    gate_percentile=args.gate_percentile, path="F0_GATE_F1",
                    device=patch_device, decomp_cache=decomp_cache,
                    artifacts_dir=artifacts_model_dir, artifact_registry=artifact_registry,
                )
                patched4.append((blk_i, bn_i, orig_i, repl_i))
            for _, _, _, repl_i in patched4:
                reset_replaced_counters(repl_i)
            y4_c = fn4()
            perf4_c, ram4_c = measure_phase_ram(
                lambda: benchmark_ns(fn4, warmup=2, iterations=7, device=device))
            q4 = (
                cosine_nrmse(y4_ref, y4_c)
                if (y4_ref is not None and y4_c is not None)
                else {"cosine": 0.0, "nrmse": 1.0}
            )
            skip_per_block = {bn_i: read_replaced_counters(repl_i) for _, bn_i, _, repl_i in patched4}
            f0_tot = sum(c["F0_calls"] for c in skip_per_block.values())
            f1_tot = sum(c["F1_calls"] for c in skip_per_block.values())
            skip4 = 1.0 - (f1_tot / max(f0_tot, 1))
            all_fulls = [f for _, _, _, repl_i in patched4 for f in repl_i]
            cand_disk4 = sum(int(artifact_registry[f]["total_bytes"]) for f in all_fulls)
            base_disk4 = sum(int(decomp_cache[f][0].baseline_bytes) for f in all_fulls)
            blocks4_pass = q4["cosine"] >= 0.98 and skip4 > 0
            recorder.emit(
                bid("BLOCKS4_GATED"),
                "PASS" if blocks4_pass else "EXPERIMENTAL_FAIL",
                scope=(
                    f"C3 passo 15: {n4} blocos consecutivos com TODAS as Linears F0({codec_name})"
                    f"+Gate·F1; forward real do modelo; qualidade na saída do bloco {watch_name}; "
                    f"RAM topo=null (registro de operação — adendo V3: RSS de topo só no "
                    f"FULLMODEL_E2E_TOKS; fases VmRSS em metrics.memory); "
                    f"artefatos F0/F1 reais em disco (os.stat)"
                ),
                quality_output=q4,
                full_gate=blocks4_pass,
                metrics={
                    "operation": {
                        "metric": "model_forward_latency_4blocks",
                        "baseline_median_ms_proxy": perf4_base["median_ms"],
                        "candidate_median_ms_proxy": perf4_c["median_ms"],
                        "speedup_x_proxy": perf4_base["median_ms"] / max(perf4_c["median_ms"], 1e-12),
                        "baseline_latency": perf4_base,
                        "candidate_latency": perf4_c,
                        "n_tokens": n_tokens_small,
                        "tok_s_note": "proxy de forward; tok/s de topo só no FULLMODEL_E2E_TOKS",
                    },
                    "memory": memory_metrics(ram4_base, ram4_c),
                    "cascade": {
                        "n_blocks_patched": n4,
                        "watch_block": watch_name,
                        "codec_f0": codec_name,
                        "F0_calls": f0_tot,
                        "F1_calls": f1_tot,
                        "F1_skip_rate": skip4,
                        "skip_rate_per_block": skip_per_block,
                        "n_linears_patched": len(all_fulls),
                        "original_weight_on_hot_path": False,
                    },
                },
                baseline_disk=base_disk4,
                candidate_disk=cand_disk4,
                primary=True,
                notes=(
                    f"Passo 15: {n4} blocos gated. cos={q4['cosine']:.4f} skip={skip4:.3f} "
                    f"ms {perf4_base['median_ms']:.1f}->{perf4_c['median_ms']:.1f} "
                    f"disco {base_disk4}->{cand_disk4} B (artefatos reais)."
                ),
                highlight=f"cos={q4['cosine']:.4f} skip={skip4:.3f}",
            )
            summary["blocks4"] = {
                "n_blocks": n4, "cosine": q4["cosine"], "nrmse": q4["nrmse"],
                "f1_skip_rate": skip4, "pass": blocks4_pass,
                "baseline_median_ms": perf4_base["median_ms"],
                "candidate_median_ms": perf4_c["median_ms"],
                "candidate_disk_bytes": cand_disk4, "baseline_disk_bytes": base_disk4,
            }
        except Exception as exc:
            traceback.print_exc()
            recorder.emit(
                bid("BLOCKS4_GATED"), "FAIL",
                scope="C3 4 blocos gated (erro em runtime)",
                metrics={"error": f"{type(exc).__name__}: {exc}"[:800]},
                notes=f"Falha no passo 15: {exc}",
                highlight="erro",
            )
        finally:
            for blk_i, bn_i, orig_i, _ in patched4:
                restore_block_linears(blk_i, orig_i, bn_i)

    # ------------------------------------------------------------------
    # Passo 16 — C3_<TECH>_FULLMODEL_E2E_TOKS (tok/s REAL baseline E candidato)
    # ------------------------------------------------------------------
    summary["fullmodel"] = {"pass": False, "skipped": False}
    if not decision_pass:
        # docs/C3_METHODOLOGY.md: passo 16 só executa com C3_<TECH>_C1_DECISION = PASS
        recorder.emit(
            bid("FULLMODEL_E2E_TOKS"), "SKIPPED",
            scope="C3 passo 16 (bloqueado pela decisão C1 reprovada)",
            metrics={"skipped": True, "c1_decision_pass": False},
            notes=(
                f"Passo 16 não executado: {bid('C1_DECISION')} = FAIL reprova a expansão "
                "(docs/C3_METHODOLOGY.md). tok/s de topo permanecem null — nenhuma "
                "métrica primária é publicada com C1 reprovado."
            ),
            highlight="SKIPPED (C1 reprovado)",
        )
        summary["fullmodel"]["skipped"] = True
        summary["fullmodel"]["skip_reason"] = "c1_decision_fail"
    elif args.skip_full_model:
        recorder.emit(
            bid("FULLMODEL_E2E_TOKS"), "SKIPPED",
            scope="C3 passo 16 (pulado por --skip-full-model)",
            metrics={"skipped": True},
            notes=(
                "Passo 16 pulado via --skip-full-model (ambiente com pouca RAM). "
                "tok/s de topo permanecem null — nenhum valor estimado é publicado."
            ),
            highlight="SKIPPED (flag)",
        )
        summary["fullmodel"]["skipped"] = True
        summary["fullmodel"]["skip_reason"] = "skip_full_model_flag"
    elif not blocks:
        recorder.emit(
            bid("FULLMODEL_E2E_TOKS"), "FAIL",
            scope="C3 passo 16 (indisponível)",
            metrics={"error": "nenhum transformer block encontrado"},
            notes="Sem blocos para patch do modelo completo.",
            highlight="sem blocos",
        )
    else:
        print(f"[C3] passo 16: generate e2e baseline vs candidato ({len(blocks)} blocos)...")
        enc_gen = tokenizer(GENERATION_PROMPT, return_tensors="pt")
        enc_gen = {k: v.to(device) for k, v in enc_gen.items()}
        logits_base = forward_logits(model, enc_gen)
        base_gen, ram_gen_base = measure_phase_ram(
            lambda: measure_generate(
                model, tokenizer, GENERATION_PROMPT, device,
                max_new_tokens=args.max_new_tokens, warmup=2, timed=3,
            )
        )
        print(f"[C3] baseline {base_gen['tok_s_median']:.2f} tok/s "
              f"({base_gen['n_new_tokens']} tokens novos, greedy)")
        patched_all: List[Tuple[nn.Module, str, Dict[str, nn.Linear], Dict[str, nn.Module]]] = []
        try:
            def _patch_all():
                for bn_i, blk_i in blocks:
                    orig_i, repl_i = patch_one_block(
                        blk_i, bn_i, codec,
                        rank=args.rank, group_size=args.group_size,
                        gate_percentile=args.gate_percentile, path="F0_GATE_F1",
                        device=patch_device, decomp_cache=decomp_cache,
                        artifacts_dir=artifacts_model_dir, artifact_registry=artifact_registry,
                    )
                    patched_all.append((blk_i, bn_i, orig_i, repl_i))
                return len(patched_all)

            _, ram_patch = measure_phase_ram(_patch_all)
            for _, _, _, repl_i in patched_all:
                reset_replaced_counters(repl_i)
            logits_cand = forward_logits(model, enc_gen)
            cand_gen, ram_gen_cand = measure_phase_ram(
                lambda: measure_generate(
                    model, tokenizer, GENERATION_PROMPT, device,
                    max_new_tokens=args.max_new_tokens, warmup=2, timed=3,
                )
            )
            print(f"[C3] candidato {cand_gen['tok_s_median']:.2f} tok/s")
            f0_all = sum(int(m.f0_calls) for _, _, _, repl_i in patched_all for m in repl_i.values())
            f1_all = sum(int(m.f1_calls) for _, _, _, repl_i in patched_all for m in repl_i.values())
            skip_all = 1.0 - (f1_all / max(f0_all, 1))
            all_fulls_fm = [f for _, _, _, repl_i in patched_all for f in repl_i]
            cand_disk_fm = sum(int(artifact_registry[f]["total_bytes"]) for f in all_fulls_fm)
            base_disk_fm = sum(int(decomp_cache[f][0].baseline_bytes) for f in all_fulls_fm)
            em = token_exact_match(base_gen["new_token_ids"], cand_gen["new_token_ids"])
            logits_q = (
                cosine_nrmse(logits_base, logits_cand)
                if (logits_base is not None and logits_cand is not None)
                else None
            )
            logits_cos = logits_q["cosine"] if logits_q else None
            baseline_tok_s = float(base_gen["tok_s_median"])
            candidate_tok_s = float(cand_gen["tok_s_median"])
            quality_ok = logits_cos is not None and logits_cos >= 0.95
            fm_pass = baseline_tok_s > 0 and candidate_tok_s > 0 and quality_ok
            mem = memory_metrics(ram_gen_base, ram_gen_cand)
            mem["patch_phase"] = ram_patch
            mem["baseline_phase_scope"] = "model.generate baseline (modelo original completo)"
            mem["candidate_phase_scope"] = "model.generate candidato (TODOS os blocos patchados)"
            recorder.emit(
                bid("FULLMODEL_E2E_TOKS"),
                "PASS" if fm_pass else "FAIL",
                scope=(
                    f"C3 passo 16: modelo completo — TODAS as Linears dos {len(blocks)} blocos "
                    f"patchadas com F0({codec_name})+Gate·F1; baseline E candidato via "
                    f"model.generate (mesmo prompt/greedy/max_new_tokens={args.max_new_tokens}; "
                    f"2 warmup + 3 medições; mediana); RAM topo=pico VmRSS por fase"
                ),
                quality_output={
                    "logits": logits_q,
                    "token_exact_match": em,
                },
                full_gate=fm_pass,
                metrics={
                    "operation": {
                        "metric": "e2e_generate_tok_s",
                        "baseline": {k: v for k, v in base_gen.items() if k != "new_token_ids"},
                        "candidate": {k: v for k, v in cand_gen.items() if k != "new_token_ids"},
                        "speedup_x": candidate_tok_s / max(baseline_tok_s, 1e-12),
                    },
                    "memory": mem,
                    "cascade": {
                        "n_blocks_patched": len(patched_all),
                        "n_linears_patched": len(all_fulls_fm),
                        "codec_f0": codec_name,
                        "F0_calls": f0_all,
                        "F1_calls": f1_all,
                        "F1_skip_rate": skip_all,
                        "original_weight_on_hot_path": False,
                        "lm_head_note": "lm_head/embeddings fora dos blocos permanecem originais",
                    },
                },
                baseline_disk=base_disk_fm,
                candidate_disk=cand_disk_fm,
                baseline_tok_s=baseline_tok_s,
                candidate_tok_s=candidate_tok_s,
                primary=True,
                ram_base=ram_gen_base,
                ram_cand=ram_gen_cand,
                notes=(
                    f"Passo 16 (coroa): baseline={baseline_tok_s:.2f} tok/s "
                    f"candidato={candidate_tok_s:.2f} tok/s (ambos REAIS, model.generate). "
                    f"logits_cos={logits_cos if logits_cos is None else round(logits_cos, 4)} "
                    f"exact_match={em['exact_match_rate']:.3f} skip={skip_all:.3f} "
                    f"disco {base_disk_fm}->{cand_disk_fm} B (artefatos reais os.stat)."
                ),
                highlight=f"{baseline_tok_s:.2f}->{candidate_tok_s:.2f} tok/s",
            )
            summary["fullmodel"] = {
                "pass": fm_pass, "skipped": False,
                "baseline_tok_s": baseline_tok_s, "candidate_tok_s": candidate_tok_s,
                "speedup_x": candidate_tok_s / max(baseline_tok_s, 1e-12),
                "logits_cosine": logits_cos,
                "token_exact_match_rate": em["exact_match_rate"],
                "f1_skip_rate": skip_all,
                "baseline_disk_bytes": base_disk_fm,
                "candidate_disk_bytes": cand_disk_fm,
            }
        except Exception as exc:
            traceback.print_exc()
            recorder.emit(
                bid("FULLMODEL_E2E_TOKS"), "FAIL",
                scope="C3 passo 16 (erro em runtime)",
                metrics={"error": f"{type(exc).__name__}: {exc}"[:800]},
                notes=f"Falha no passo 16: {exc}",
                highlight="erro",
            )
        finally:
            for blk_i, bn_i, orig_i, _ in patched_all:
                restore_block_linears(blk_i, orig_i, bn_i)
            print("[C3] modelo restaurado (unpatch de todos os blocos)")

    # ------------------------------------------------------------------
    # Gain report + tabela final PT-BR
    # ------------------------------------------------------------------
    summary["generated_at"] = utc()
    summary["records_total"] = len(recorder.summary_rows)
    summary["records_pass"] = sum(1 for r in recorder.summary_rows if r["status"] == "PASS")
    gain_path = out_dir / f"c3_{tech}_gain_report.json"
    gain_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print()
    print("=" * 96)
    print(f"C3 {tech_upper} — metodologia de 16 passos | codec={codec_name} | model={model_id} | device={device.type}")
    print("=" * 96)
    print(f"{'battery_id':<44} {'status':<18} destaque")
    print("-" * 96)
    for row in recorder.summary_rows:
        print(f"{row['battery_id']:<44} {row['status']:<18} {row['highlight']}")
    print("-" * 96)
    print(f"Decisão C1              : {'APROVADO' if summary.get('c1_decision_pass') else 'REPROVADO'}")
    fm = summary.get("fullmodel") or {}
    if fm.get("skipped"):
        _reason = (
            "decisão C1 reprovada" if fm.get("skip_reason") == "c1_decision_fail"
            else "--skip-full-model"
        )
        print(f"Tok/s e2e               : SKIPPED ({_reason})")
    elif fm.get("baseline_tok_s") is not None:
        print(
            f"Tok/s e2e (REAL)        : baseline={fm['baseline_tok_s']:.2f} "
            f"candidato={fm['candidate_tok_s']:.2f} ({fm['speedup_x']:.2f}x)"
        )
    print(f"Baterias JSON           : {recorder.json_path}")
    print(f"Gain report             : {gain_path}")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    _rc = 0
    try:
        _rc = main() or 0
    except SystemExit as _e:
        _rc = int(_e.code) if isinstance(_e.code, int) else 0
    except Exception:
        traceback.print_exc()
        _rc = 0  # baterias reportam; só erro de infraestrutura já foi registrado
    finally:
        try:
            cleanup_colab_workspace(label="C3-METHODOLOGY", wipe_hf_cache=False)
        except Exception as _ce:
            print(f"[cleanup] AVISO: {_ce}")
    raise SystemExit(_rc)
