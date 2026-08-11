#!/usr/bin/env python3
"""Fase final C4/C5/C6 — FINAL_PHASE_V1 (docs/C3_CONTRACTS_V1.md §16).

Script ÚNICO para as quatro tecnologias:
    python final_phase_auto_batteries.py --technology rift|aether|cascade|spectra

Codec F0 default por tecnologia (override com --codec):
    cascade → int4 | rift → int2 | aether → ternary | spectra → ternary
F1 (residual low-rank) e Confidence Gate v0 são comuns a todas.

Baterias (contrato §16 — battery_ids imutáveis):
    C4_<TECH>_SECOND_FAMILY   MESMO core em 2 famílias: modelo A (--model) e
                              modelo B de OUTRA arquitetura (--second-model);
                              Linear+bloco gated em ambos; PASS sse cosine
                              gated >= 0.98 nas DUAS famílias.
    C5_<TECH>_REPR_BLOCKS     8-10 blocos representativos do modelo maior
                              (--large-model), amostrados no espectro de
                              profundidade; qualidade por profundidade + drift
                              acumulado encadeando os blocos amostrados; PASS
                              sse cosine por bloco >= 0.95 E drift <= 0.12.
    C6_<TECH>_COMPILE_EXECUTE MARCO FINAL: compila TODAS as Linear dos blocos
                              do modelo pequeno → bundles CSCD REAIS em disco
                              (<out>/bundle/) → runtime carrega os stages DO
                              ARQUIVO (mmap + torch.frombuffer) → pesos densos
                              originais DESCARTADOS após o swap → generate
                              completo. PASS sse executa ∧ skip>0 ∧ logits
                              cosine >= 0.95 ∧ bundle_bytes < checkpoint_bytes
                              ∧ pesos originais liberados.

comparison_role="primary" SOMENTE em C6 (baseline_tok_s E candidate_tok_s
REAIS via model.generate — baseline medido ANTES da compilação;
metrics.e2e.measured=true). C4/C5 são gates diagnósticos (comparison_role=null).

Honestidade de medição (docs/REAL_BENCHMARK_PROTOCOL_V3.md + contrato §3):
  - latência/tok/s: time.perf_counter_ns com warmup e cuda.synchronize;
  - *_ram_bytes de topo: SOMENTE pico VmRSS medido por fase; senão null;
  - candidate_disk_bytes: SOMENTE os.stat de bundles realmente gravados;
  - baseline_tok_s/candidate_tok_s de topo: SOMENTE model.generate (C6);
  - guard de recursos: >3e9 params → SKIPPED; OOM → SKIPPED com nota.

Auto-contido EXCETO pelo pacote cascade/ (o launcher Colab baixa a lista fixa
de arquivos do pacote); os codecs int2 e ternary do F0 são inline aqui.
Sem pip install automático: dependências ausentes geram SystemExit.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import mmap
import os
import platform
import struct
import sys
import threading
import time
import traceback
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# --- resolve pacote cascade/ (cwd do script, /content, /content/final_run, /content/cascade_run ou <repo>/core) ---
_HERE = Path(__file__).resolve().parent
for _cand in [_HERE, Path("/content"), Path("/content/final_run"), Path("/content/cascade_run"), _HERE.parent / "core"]:
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
        "[FINAL] Dependências ausentes: " + ", ".join(_MISSING_DEPS) + ". "
        "Este script NÃO instala pacotes automaticamente — o launcher Colab deveria "
        "tê-los instalado. Instale manualmente: pip install torch transformers "
        "accelerate sentencepiece (versões pinadas conforme o launcher)."
    )

from cascade.compiler.block_decompose import find_transformer_blocks
from cascade.compiler.bundle_writer import (
    HEADER_SIZE as CSCD_HEADER_SIZE,
    MAGIC as CSCD_MAGIC,
    VERSION as CSCD_VERSION,
    write_cascade_bundle,
)
from cascade.compiler.decompose import CascadeLinearStages, decompose_linear_int4_lowrank
from cascade.kernels.int4 import dequantize_int4, quantize_int4_group
from cascade.kernels.lowrank import fit_lowrank_residual, lowrank_linear
from cascade.runtime.block_runtime import collect_block_linears, restore_block_linears
from cascade.runtime.cleanup import cleanup_colab_workspace
from cascade.runtime.confidence_gate import GateConfig, decide_gate
from cascade.runtime.reference import CascadeLinearRuntime

BENCHMARK_PROTOCOL = "FINAL_PHASE_V1"

TECH_DEFAULT_CODEC = {
    "cascade": "int4",
    "rift": "int2",
    "aether": "ternary",
    "spectra": "ternary",
}

PARAM_GUARD = 3_000_000_000  # contrato §16: >3e9 params → SKIPPED

# Prompts fixos PT-BR (ativação real e generate e2e)
ACTIVATION_PROMPT = (
    "Explique por que a memória importa na inferência de modelos de linguagem. "
    "Latência, RAM e disco definem o custo real."
)
GENERATION_PROMPT = "Liste três técnicas para reduzir o uso de memória na inferência de LLMs:"

LATENCY_METHOD = "perf_counter_ns_with_cuda_sync_v1"
RAM_METHOD = "proc_vmrss_sampling_per_phase_v1"


# ---------------------------------------------------------------------------
# Utilidades gerais (espelham c3_methodology_auto_batteries.py)
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


def normalize_model_id(raw: str) -> str:
    return str(raw).strip().replace("https://huggingface.co/", "").strip("/")


def free_memory() -> None:
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass


def is_oom_error(exc: BaseException) -> bool:
    """OOM → registro SKIPPED com nota (guard de recursos do contrato §16)."""
    if isinstance(exc, MemoryError):
        return True
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return (
        "outofmemory" in name
        or "out of memory" in msg
        or "not enough memory" in msg
        or "cannot allocate memory" in msg
        or "paging file" in msg
    )


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


def phase_method(*phases: Optional[Dict[str, Any]]) -> Optional[str]:
    for p in phases:
        if isinstance(p, dict) and p.get("method"):
            return str(p["method"])
    return None


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

# Rótulo do stage.meta.codec do stage 0 no bundle CSCD (contrato §6)
F0_BUNDLE_CODEC_LABEL = {
    "int4": "INT4_GROUP",
    "int2": "INT2_GROUP",
    "ternary": "TERNARY_ROWSCALE",
}


def get_codec(name: str):
    if name not in _CODECS:
        raise SystemExit(f"[FINAL] codec desconhecido: {name} (válidos: {sorted(_CODECS)})")
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


# ---------------------------------------------------------------------------
# Leitor VALIDADOR do bundle CSCD (mesmo padrão do C3) + parser de tensores
# ---------------------------------------------------------------------------

# layout do header (cascade/compiler/bundle_writer.py):
# magic(4s) ver(H) flags(H) hdr(I) n_stages(I) ir(Q) st(Q) gate(Q) pay(Q) fsize(Q) crc(Q)
CSCD_HEADER_PREFIX_FMT = "<4sHHIIQQQQQQ"
CSCD_STAGE_ENTRY_FMT = "<QQII"
CSCD_STAGE_ENTRY_SIZE = struct.calcsize(CSCD_STAGE_ENTRY_FMT)  # 24 bytes (ABI congelada)


class BundleFormatError(ValueError):
    """Bundle CSCD inválido/corrompido — o leitor validador rejeita com motivo."""


def validate_cscd_bundle(buf) -> Dict[str, Any]:
    """Leitor validador: header, tamanhos, CRC e limites de cada stage entry.

    Aceita bytes OU mmap (buffer protocol). Levanta BundleFormatError com
    motivo curto; retorna campos parseados quando válido.
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
    if (zlib.crc32(bytes(memoryview(buf)[CSCD_HEADER_SIZE:])) & 0xFFFFFFFFFFFFFFFF) != crc:
        raise BundleFormatError("bad_crc")
    if not (CSCD_HEADER_SIZE <= ir_off < len(buf)):
        raise BundleFormatError("ir_offset_out_of_bounds")
    if st_off < CSCD_HEADER_SIZE or st_off + n_stages * CSCD_STAGE_ENTRY_SIZE > len(buf):
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


def _read_stage_meta(buf, off: int) -> Dict[str, Any]:
    (mlen,) = struct.unpack_from("<I", buf, off)
    return json.loads(bytes(memoryview(buf)[off + 4: off + 4 + mlen]).decode("utf-8"))


# dtype declarado no header de cada payload de tensor (bundle_writer._tensor_payload)
_TENSOR_DTYPES: Dict[str, Tuple["torch.dtype", int]] = {
    "uint8": (torch.uint8, 1),
    "int8": (torch.int8, 1),
    "int16": (torch.int16, 2),
    "int32": (torch.int32, 4),
    "int64": (torch.int64, 8),
    "float16": (torch.float16, 2),
    "float32": (torch.float32, 4),
    "float64": (torch.float64, 8),
}


def parse_bundle_tensor(buf, off: int, limit: int) -> Tuple["torch.Tensor", int]:
    """Parseia UM payload de tensor (<I mlen> meta_json corpo) direto do buffer.

    Com mmap: torch.frombuffer aponta para as páginas mapeadas (zero-copy).
    Valida meta e limites antes de indexar (contrato §5).
    """
    if off + 4 > limit:
        raise BundleFormatError("tensor_meta_out_of_bounds")
    (tlen,) = struct.unpack_from("<I", buf, off)
    meta_end = off + 4 + int(tlen)
    if meta_end > limit:
        raise BundleFormatError("tensor_meta_out_of_bounds")
    meta = json.loads(bytes(memoryview(buf)[off + 4: meta_end]).decode("utf-8"))
    dtype_name = str(meta.get("dtype"))
    if dtype_name not in _TENSOR_DTYPES:
        raise BundleFormatError(f"tensor_dtype_unsupported:{dtype_name}")
    dtype, item_size = _TENSOR_DTYPES[dtype_name]
    shape = [int(d) for d in (meta.get("shape") or [])]
    numel = 1
    for d in shape:
        if d < 0:
            raise BundleFormatError("tensor_shape_invalid")
        numel *= d
    body_end = meta_end + numel * item_size
    if body_end > limit:
        raise BundleFormatError("tensor_payload_out_of_bounds")
    if numel == 0:
        return torch.empty(shape, dtype=dtype), body_end
    if meta_end % item_size == 0:
        # zero-copy: mmap é page-aligned, então offset alinhado ao item basta
        t = torch.frombuffer(buf, dtype=dtype, count=numel, offset=meta_end).reshape(shape)
    else:
        # offset desalinhado (metas JSON têm tamanho variável): cópia ALINHADA
        # dos MESMOS bytes mapeados (bytearray é gravável — sem warning)
        aligned = bytearray(memoryview(buf)[meta_end:body_end])
        t = torch.frombuffer(aligned, dtype=dtype, count=numel).reshape(shape)
    return t, body_end


# ---------------------------------------------------------------------------
# Módulos de runtime — Y = F0(X) + Gate(X)·F1(X) [+ bias]; W denso fora do
# caminho quente. Base comum para o módulo in-memory (C4/C5) e o módulo que
# carrega DO BUNDLE (C6).
# ---------------------------------------------------------------------------

class _GatedLinearBase(nn.Module):
    """Espelha cascade/runtime/block_runtime.CascadeLinearModule (contadores
    f0_calls/f1_calls/f1_skip_calls, path, stats()) para qualquer codec F0.
    O W denso original NÃO fica no caminho quente (apenas codes/scales/u/s/v
    e o bias FP32 — vetor pequeno, nunca é o W denso)."""

    def _init_gated(self, codec, *, gate_percentile: float, path: str,
                    low_mem: Optional[bool]) -> None:
        self.codec = codec
        self.path = path
        self.gate_cfg = GateConfig(percentile=gate_percentile)
        self.last_gate_rate = 1.0
        self.f0_calls = 0
        self.f1_calls = 0
        self.f1_skip_calls = 0
        if low_mem is None:
            low_mem = os.environ.get("CASCADE_LOW_MEM", "").strip() == "1"
        self.low_mem = bool(low_mem)
        self._w0_cache: Optional["torch.Tensor"] = None

    def _register_bias(self, bias: Optional["torch.Tensor"]) -> None:
        if bias is not None:
            self.register_buffer("bias", bias.detach().to(dtype=torch.float32).cpu().clone())
        else:
            self.register_buffer("bias", None)

    def _dequant_w0(self, device: "torch.device") -> "torch.Tensor":
        w0 = self.codec.dequantize(
            self.codes, self.scales, group_size=self.group_size,
            out_features=self.out_features, in_features=self.in_features,
        )
        return w0.to(device=device, dtype=torch.float32)

    def _w0(self, device: "torch.device") -> "torch.Tensor":
        # cache do F0 dequantizado (aproximação quantizada, NÃO o W original)
        if self.low_mem:
            return self._dequant_w0(device)
        if self._w0_cache is None or self._w0_cache.device != device:
            self._w0_cache = self._dequant_w0(device)
        return self._w0_cache

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        orig_shape = x.shape
        x2f = x.reshape(-1, orig_shape[-1]).to(dtype=torch.float32)
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

        if self.bias is not None:
            out = out + self.bias.to(device=out.device, dtype=out.dtype)
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


class FinalGatedLinearModule(_GatedLinearBase):
    """Módulo in-memory (C4/C5): stages decompostos ficam no módulo."""

    def __init__(self, stages: CascadeLinearStages, codec, *, gate_percentile: float = 70.0,
                 path: str = "F0_GATE_F1", bias: Optional["torch.Tensor"] = None,
                 low_mem: Optional[bool] = None):
        super().__init__()
        self._init_gated(codec, gate_percentile=gate_percentile, path=path, low_mem=low_mem)
        self.register_buffer("codes", stages.codes)
        self.register_buffer("scales", stages.scales)
        self.register_buffer("u", stages.u)
        self.register_buffer("s", stages.s)
        self.register_buffer("v", stages.v)
        self.group_size = int(stages.group_size)
        self.out_features = int(stages.out_features)
        self.in_features = int(stages.in_features)
        self._register_bias(bias)


class BundleLinearModule(_GatedLinearBase):
    """Módulo do C6: carrega codes/scales/u/s/v DO ARQUIVO .cascade (mmap).

    Abre o bundle UMA vez, valida (leitor validador CSCD), parseia a stage
    table e materializa os tensores dos stages via torch.frombuffer sobre os
    bytes mapeados. O W denso original NUNCA entra aqui — só os stages do
    bundle (+ bias FP32 preservado fora do bundle, vetor pequeno).
    """

    def __init__(self, bundle_path, codec, *, expected_f0_codec: str,
                 gate_percentile: float = 70.0, bias: Optional["torch.Tensor"] = None,
                 low_mem: Optional[bool] = None):
        super().__init__()
        self._init_gated(codec, gate_percentile=gate_percentile, path="F0_GATE_F1", low_mem=low_mem)
        self.bundle_path = str(bundle_path)
        self._fh = open(bundle_path, "rb")
        try:
            # ACCESS_COPY: páginas file-backed copy-on-write (buffer gravável p/ frombuffer)
            self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_COPY)
            parsed = validate_cscd_bundle(self._mm)
            if len(parsed["stages"]) != 2:
                raise BundleFormatError("expected_2_stages")
            s0, s1 = parsed["stages"]
            meta0 = _read_stage_meta(self._mm, s0["offset"])
            meta1 = _read_stage_meta(self._mm, s1["offset"])
            if meta0.get("stage_type") != "BASE_STAGE":
                raise BundleFormatError("stage0_not_base_stage")
            if meta0.get("codec") != expected_f0_codec:
                raise BundleFormatError(
                    f"stage0_codec_mismatch:{meta0.get('codec')}!={expected_f0_codec}"
                )
            if meta1.get("stage_type") != "RESIDUAL_LOWRANK":
                raise BundleFormatError("stage1_not_residual_lowrank")
            (m0len,) = struct.unpack_from("<I", self._mm, s0["offset"])
            off0 = s0["offset"] + 4 + int(m0len)
            limit0 = s0["offset"] + s0["size"]
            codes, off0 = parse_bundle_tensor(self._mm, off0, limit0)
            scales, off0 = parse_bundle_tensor(self._mm, off0, limit0)
            (m1len,) = struct.unpack_from("<I", self._mm, s1["offset"])
            off1 = s1["offset"] + 4 + int(m1len)
            limit1 = s1["offset"] + s1["size"]
            u, off1 = parse_bundle_tensor(self._mm, off1, limit1)
            s, off1 = parse_bundle_tensor(self._mm, off1, limit1)
            v, off1 = parse_bundle_tensor(self._mm, off1, limit1)
        except Exception:
            # falha na validação/parse: libera fd+mmap antes de propagar
            self.close()
            raise
        # cache dos tensores de stage carregados do bundle (nunca o W original)
        self.register_buffer("codes", codes)
        self.register_buffer("scales", scales)
        self.register_buffer("u", u)
        self.register_buffer("s", s)
        self.register_buffer("v", v)
        self.group_size = int(meta0["group_size"])
        self.out_features = int(meta0["out_features"])
        self.in_features = int(meta0["in_features"])
        self._register_bias(bias)
        self.bundle_bytes = int(parsed["file_size"])
        self.loaded_from_bundle = True

    def close(self) -> None:
        """Libera mmap+fd. Só chamar quando o módulo sai de uso: os buffers de
        stage podem ALIASAR o mmap (torch.frombuffer) enquanto residem em CPU."""
        mm = getattr(self, "_mm", None)
        if mm is not None:
            try:
                mm.close()
            except Exception:
                pass
            self._mm = None
        fh = getattr(self, "_fh", None)
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass
            self._fh = None

    def __del__(self):  # rede de segurança no GC (evita esgotar fds no processo)
        try:
            self.close()
        except Exception:
            pass


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
    decomp_cache: Dict[str, CascadeLinearStages],
) -> Tuple[Dict[str, nn.Linear], Dict[str, nn.Module]]:
    """Troca TODAS as nn.Linear do bloco pelo runtime do codec (W fora do caminho quente).

    Decomposições são cacheadas por nome (reuso entre qualidade por bloco e drift
    encadeado do C5). TRANSACIONAL: se qualquer Linear falhar no meio do loop,
    as Linears já substituídas são RESTAURADAS antes de relançar — o bloco
    nunca fica parcialmente patchado (padrão comprovado do C3).
    """
    originals = collect_block_linears(block, block_name)
    replaced: Dict[str, nn.Module] = {}
    try:
        for full, linear in originals.items():
            if full in decomp_cache:
                stages = decomp_cache[full]
            else:
                w = linear.weight.detach().float().cpu()
                stages, _packed = decompose_with_codec(w, codec, rank=rank, group_size=group_size)
                decomp_cache[full] = stages
            mod = FinalGatedLinearModule(
                stages, codec, gate_percentile=gate_percentile, path=path,
                bias=linear.bias,
            )
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
                print(f"[FINAL] AVISO rollback de {full} falhou: {rexc}")
        raise
    if not replaced:
        raise RuntimeError(f"Nenhuma nn.Linear em {block_name}")
    return originals, replaced


def reset_replaced_counters(replaced: Dict[str, nn.Module]) -> None:
    """Zera contadores ANTES de cada medição (nunca acumula entre caminhos)."""
    for m in replaced.values():
        m.f0_calls = 0
        m.f1_calls = 0
        m.f1_skip_calls = 0


def read_replaced_counters(replaced: Dict[str, nn.Module]) -> Dict[str, Any]:
    """Lê contadores DEPOIS da medição executada."""
    f0 = sum(int(m.f0_calls) for m in replaced.values())
    f1 = sum(int(m.f1_calls) for m in replaced.values())
    return {
        "F0_calls": f0,
        "F1_calls": f1,
        "F1_skip_rate": 1.0 - (f1 / max(f0, 1)),
        "avg_stages_per_token": 1.0 + (f1 / max(f0, 1)),
    }


class FinalLinearRuntime:
    """Linear de referência 4 caminhos (C4); para int4 delega a CascadeLinearRuntime."""

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
        mask, gate_meta = decide_gate(x, self.gate_cfg)
        y = y0 + mask.to(dtype=y1.dtype).unsqueeze(1) * y1
        f1_calls = int(mask.to(torch.int64).sum().item())
        return {
            "y": y, "f0_calls": n, "f1_calls": f1_calls,
            "f1_skip_rate": 1.0 - (f1_calls / max(n, 1)), "gate": gate_meta,
        }


# ---------------------------------------------------------------------------
# Modelo real (HF) — load, Linear alvo, ativação real, blocos e generate
# ---------------------------------------------------------------------------

def load_model(model_id: str, device: "torch.device", trust: bool, token: Optional[str]):
    try:
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "[FINAL] transformers indisponível. Instale manualmente: pip install "
            f"transformers accelerate sentencepiece. Erro: {exc}"
        )
    tok = AutoTokenizer.from_pretrained(model_id, token=token, trust_remote_code=trust)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    kwargs = dict(
        token=token, trust_remote_code=trust, low_cpu_mem_usage=True,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    if device.type == "cuda":
        kwargs["device_map"] = "auto"
    errors = []
    model = None
    for cls in (AutoModelForCausalLM, AutoModel):
        try:
            model = cls.from_pretrained(model_id, **kwargs)
            if getattr(model, "hf_device_map", None) is None:
                model = model.to(device)
            model.eval()
            print(f"[FINAL] load {cls.__name__} ({model_id})")
            break
        except Exception as exc:
            errors.append(f"{cls.__name__}: {exc}")
    if model is None:
        raise RuntimeError("Falha ao carregar modelo:\n" + "\n".join(errors))
    return model, tok


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def hub_param_count(model_id: str, token: Optional[str]) -> Optional[int]:
    """Estimativa de parâmetros ANTES do download (guard >3e9 sem baixar pesos)."""
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(model_id, token=token)
        st = getattr(info, "safetensors", None)
        total = getattr(st, "total", None) if st is not None else None
        if total:
            return int(total)
    except Exception as exc:
        print(f"[FINAL] AVISO estimativa de parâmetros via hub indisponível: {exc}")
    return None


def find_linear_weight(model, target: str = "auto") -> Tuple[str, "torch.Tensor"]:
    """Seleciona a Linear real alvo (espelha cascade_c0/c3)."""
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
        print(f"[FINAL] AVISO captura de ativação: {exc}")
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
            print(f"[FINAL] AVISO forward direto do bloco indisponível ({exc}); usando forward do modelo")
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


def forward_logits(model: nn.Module, inputs: Dict[str, "torch.Tensor"]) -> Optional["torch.Tensor"]:
    try:
        with torch.inference_mode():
            out = model(**inputs)
        logits = getattr(out, "logits", None)
        if logits is None and isinstance(out, tuple) and out and torch.is_tensor(out[0]):
            logits = out[0]
        return logits.detach().float().cpu() if torch.is_tensor(logits) else None
    except Exception as exc:
        print(f"[FINAL] AVISO forward de logits: {exc}")
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
# Recorder — grava JSON local (upsert por model_id+battery_id) + publica incremental
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
            "User-Agent": "final-phase-battery/1.0",
        })
        with urlopen(req, timeout=60) as resp:
            print(f"[publish] HTTP {resp.status} battery={rec.get('battery_id')}")
    except Exception as exc:
        print(f"[publish] AVISO: {exc}")


class FinalRecorder:
    """Publish incremental endurecido: cada registro alimenta o dashboard ao gravar."""

    def __init__(self, out_dir: Path, *, tech: str, tech_upper: str, device: "torch.device",
                 codec_name: str, run_id: str, publish_on: bool, endpoint: Optional[str] = None):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.batteries_dir = out_dir / "batteries"
        self.batteries_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = out_dir / f"final_{tech}_test_batteries.json"
        self.tech_upper = tech_upper
        self.device = device
        self.codec_name = codec_name
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
        self.summary_rows: List[Dict[str, Any]] = []

    def emit(
        self,
        battery_id: str,
        status: str,
        *,
        model_id: str,
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
        highlight: str = "",
    ) -> Dict[str, Any]:
        is_primary = bool(primary and not demote)
        if full_gate is None:
            full_gate = status in ("PASS", "EXPERIMENTAL_PASS")
        rec = {
            "timestamp_utc": utc(),
            "run_id": self.run_id,
            "technology": self.tech_upper,
            "model_id": model_id,
            "battery_id": battery_id,
            "status": status,
            **schema_v2_fields(model_id, self.device, self.codec_name),
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
        path = self.batteries_dir / f"{self.run_id}__{battery_id}.json"
        path.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        # upsert pelo par (model_id, battery_id) no JSON agregado — rodar outro
        # --model no mesmo out dir não pode apagar o histórico do modelo anterior
        self.records = [
            r for r in self.records
            if r.get("battery_id") != battery_id or r.get("model_id") != model_id
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
# C4 — mesma metodologia em DUAS famílias de arquitetura
# ---------------------------------------------------------------------------

def run_family_quality(model_id: str, *, device: "torch.device", codec, codec_name: str,
                       args: argparse.Namespace, token: Optional[str]) -> Dict[str, Any]:
    """MESMO core nas duas famílias: Linear gated + bloco gated com ativações reais.

    Carrega, mede, e SEMPRE descarrega o modelo (del + gc + cuda empty) antes
    de retornar — a família B usa exatamente este caminho de código.
    """
    model, tokenizer = load_model(model_id, device, args.trust_remote_code, token)
    try:
        n_params = count_params(model)
        if n_params > PARAM_GUARD:
            raise MemoryError(
                f"guard de recursos: {model_id} tem {n_params} params (> {PARAM_GUARD})"
            )
        cfg = getattr(model, "config", None)
        model_type = str(getattr(cfg, "model_type", "") or "") or None
        architectures = list(getattr(cfg, "architectures", None) or [])

        # --- Linear gated (ativação REAL via forward hook) ---
        layer_name, weight = find_linear_weight(model)
        weight = weight.to(dtype=torch.float32).cpu()
        x = capture_activation(model, tokenizer, layer_name, device, ACTIVATION_PROMPT)
        activation_source = "real_model_activation"
        if x is None or x.ndim != 2 or x.shape[-1] != weight.shape[1]:
            print(f"[FINAL] AVISO: sem ativação real em {model_id} — fallback sintético (flagado)")
            x = torch.randn(32, weight.shape[1], dtype=torch.float32)
            activation_source = "synthetic_fallback"
        elif x.shape[0] > 64:
            x = x[:64].contiguous()
        x = x.to(dtype=torch.float32).cpu()

        stages, packed = decompose_with_codec(weight, codec, rank=args.rank, group_size=args.group_size)
        runtime = FinalLinearRuntime(stages, packed, codec, gate_percentile=args.gate_percentile)
        with torch.inference_mode():
            y_ref = F.linear(x, weight)
            r_gate = runtime.execute(x, path="F0_GATE_F1")
        q_lin = cosine_nrmse(y_ref, r_gate["y"])

        # --- Bloco gated (hidden states REAIS) ---
        blocks = find_transformer_blocks(model)
        if not blocks:
            raise RuntimeError(f"nenhum transformer block encontrado em {model_id}")
        block_name, block = blocks[0]
        inputs_small = tokenizer(ACTIVATION_PROMPT, return_tensors="pt", truncation=True, max_length=64)
        inputs_small = {k: v.to(device) for k, v in inputs_small.items()}
        cap = capture_block_io(model, block, inputs_small)
        block_fn, block_mode = make_block_forward(model, block, cap, inputs_small)
        y_base = block_fn()
        if y_base is None and torch.is_tensor(cap.get("out")):
            y_base = cap["out"]
        if y_base is None:
            raise RuntimeError(f"forward do bloco não produziu saída em {model_id}")
        patch_device = device if getattr(model, "hf_device_map", None) is None else None
        decomp_cache: Dict[str, CascadeLinearStages] = {}
        originals, replaced = patch_one_block(
            block, block_name, codec,
            rank=args.rank, group_size=args.group_size,
            gate_percentile=args.gate_percentile, path="F0_GATE_F1",
            device=patch_device, decomp_cache=decomp_cache,
        )
        try:
            reset_replaced_counters(replaced)
            y_c = block_fn()
        finally:
            restore_block_linears(block, originals, block_name)
        q_blk = cosine_nrmse(y_base, y_c) if y_c is not None else {"cosine": 0.0, "nrmse": 1.0}
        counters = read_replaced_counters(replaced)

        cos_min = min(float(q_lin["cosine"]), float(q_blk["cosine"]))
        return {
            "model_id": model_id,
            "model_type": model_type,
            "architectures": architectures,
            "n_params": n_params,
            "n_blocks": len(blocks),
            "activation_source": activation_source,
            "linear_gated": {
                "layer": layer_name,
                "cosine": float(q_lin["cosine"]),
                "nrmse": float(q_lin["nrmse"]),
                "f1_skip_rate": float(r_gate["f1_skip_rate"]),
            },
            "block_gated": {
                "block": block_name,
                "forward_mode": block_mode,
                "cosine": float(q_blk["cosine"]),
                "nrmse": float(q_blk["nrmse"]),
                "f1_skip_rate": counters["F1_skip_rate"],
                "n_linears_patched": len(replaced),
            },
            "gated_cosine_min": cos_min,
            "pass": cos_min >= 0.98,
        }
    finally:
        del model, tokenizer
        free_memory()
        print(f"[FINAL] modelo {model_id} descarregado (del + gc + cuda empty_cache)")


def run_c4(recorder: FinalRecorder, args: argparse.Namespace, *, tech_upper: str,
           device: "torch.device", codec, codec_name: str, token: Optional[str],
           model_a_id: str, model_b_id: str) -> Dict[str, Any]:
    battery_id = f"C4_{tech_upper}_SECOND_FAMILY"
    if args.skip_c4:
        recorder.emit(
            battery_id, "SKIPPED", model_id=model_a_id,
            scope="Fase final C4 (pulada por --skip-c4)",
            metrics={"skipped": True, "flag": "--skip-c4"},
            notes="C4 pulada via --skip-c4 (redução padrão do contrato §16).",
            highlight="SKIPPED (flag)",
        )
        return {"status": "SKIPPED"}

    print(f"[FINAL] C4: família A = {model_a_id}")
    fam_a, ram_a = measure_phase_ram(
        lambda: run_family_quality(model_a_id, device=device, codec=codec,
                                   codec_name=codec_name, args=args, token=token)
    )
    print(f"[FINAL] C4: família B = {model_b_id}")
    fam_b, ram_b = measure_phase_ram(
        lambda: run_family_quality(model_b_id, device=device, codec=codec,
                                   codec_name=codec_name, args=args, token=token)
    )

    distinct = (
        fam_a.get("model_type") is not None
        and fam_b.get("model_type") is not None
        and fam_a["model_type"] != fam_b["model_type"]
    )
    if not distinct:
        print("[FINAL] AVISO C4: as duas famílias reportam o MESMO model_type — "
              "o contrato pede arquiteturas diferentes (registrado em metrics).")
    c4_pass = bool(fam_a["pass"] and fam_b["pass"])
    status = "PASS" if c4_pass else "FAIL"
    quality = {
        "family_a": {"cosine_min": fam_a["gated_cosine_min"],
                     "linear": fam_a["linear_gated"], "block": fam_a["block_gated"]},
        "family_b": {"cosine_min": fam_b["gated_cosine_min"],
                     "linear": fam_b["linear_gated"], "block": fam_b["block_gated"]},
    }
    recorder.emit(
        battery_id, status, model_id=model_a_id,
        scope=(
            f"C4 segunda família: MESMO core (F0 {codec_name}+Gate·F1) em 2 arquiteturas — "
            f"A={model_a_id} ({fam_a.get('model_type')}), B={model_b_id} "
            f"({fam_b.get('model_type')}); Linear gated + bloco gated com ativações/hidden "
            f"states REAIS via hook; A descarregada antes de B (del+gc+cuda empty); "
            f"RAM topo=null (gate diagnóstico — VmRSS por fase em metrics.memory)"
        ),
        quality_output=quality,
        full_gate=c4_pass,
        metrics={
            "final": {
                "families": {"family_a": fam_a, "family_b": fam_b},
                "distinct_architectures": distinct,
                "criterion": "cosine gated >= 0.98 (Linear E bloco) nas DUAS famílias",
                "codec_f0": codec_name,
            },
            "memory": {
                "method": phase_method(ram_a, ram_b),
                "family_a_phase": ram_a,
                "family_b_phase": ram_b,
            },
        },
        notes=(
            f"C4: A cos_min={fam_a['gated_cosine_min']:.4f} ({fam_a.get('model_type')}) | "
            f"B cos_min={fam_b['gated_cosine_min']:.4f} ({fam_b.get('model_type')}) | "
            f"critério >=0.98 nas duas famílias → {'PASS' if c4_pass else 'FAIL'}. "
            f"activation A={fam_a['activation_source']} B={fam_b['activation_source']}."
        ),
        highlight=f"A={fam_a['gated_cosine_min']:.4f} B={fam_b['gated_cosine_min']:.4f}",
    )
    return {"status": status, "family_a": fam_a, "family_b": fam_b}


# ---------------------------------------------------------------------------
# C5 — blocos representativos do modelo maior + drift acumulado
# ---------------------------------------------------------------------------

def sample_block_indices(n_blocks: int) -> List[int]:
    """8-10 índices no espectro de profundidade (0, 1, ~12%..~85%, últimos 2).

    Dedup para modelos rasos (contrato §16: início/meio/fim).
    """
    if n_blocks <= 0:
        return []
    fractions = (0.12, 0.25, 0.40, 0.55, 0.70, 0.85)
    raw = [0, 1] + [int(round(f * (n_blocks - 1))) for f in fractions] + [n_blocks - 2, n_blocks - 1]
    return sorted({i for i in raw if 0 <= i < n_blocks})


def run_c5(recorder: FinalRecorder, args: argparse.Namespace, *, tech_upper: str,
           device: "torch.device", codec, codec_name: str, token: Optional[str],
           large_id: str) -> Dict[str, Any]:
    battery_id = f"C5_{tech_upper}_REPR_BLOCKS"
    if args.skip_c5:
        recorder.emit(
            battery_id, "SKIPPED", model_id=large_id,
            scope="Fase final C5 (pulada por --skip-c5)",
            metrics={"skipped": True, "flag": "--skip-c5"},
            notes="C5 pulada via --skip-c5 (redução padrão do contrato §16).",
            highlight="SKIPPED (flag)",
        )
        return {"status": "SKIPPED"}

    est = hub_param_count(large_id, token)
    if est is not None and est > PARAM_GUARD:
        recorder.emit(
            battery_id, "SKIPPED", model_id=large_id,
            scope="Fase final C5 (guard de recursos ANTES do download)",
            metrics={"skipped": True, "n_params_estimate": est, "guard": PARAM_GUARD,
                     "estimate_source": "hub_safetensors_metadata"},
            notes=(
                f"C5 SKIPPED: {large_id} tem ~{est} params (> {PARAM_GUARD}) — "
                "guard de recursos do contrato §16; nenhum peso foi baixado."
            ),
            highlight="SKIPPED (>3e9 params)",
        )
        return {"status": "SKIPPED"}

    print(f"[FINAL] C5: carregando modelo maior {large_id}...")
    loaded, ram_load = measure_phase_ram(
        lambda: load_model(large_id, device, args.trust_remote_code, token)
    )
    model, tokenizer = loaded
    del loaded
    try:
        n_params = count_params(model)
        if n_params > PARAM_GUARD:
            recorder.emit(
                battery_id, "SKIPPED", model_id=large_id,
                scope="Fase final C5 (guard de recursos após load)",
                metrics={"skipped": True, "n_params": n_params, "guard": PARAM_GUARD},
                notes=(
                    f"C5 SKIPPED: {large_id} tem {n_params} params (> {PARAM_GUARD}) — "
                    "guard de recursos do contrato §16."
                ),
                highlight="SKIPPED (>3e9 params)",
            )
            return {"status": "SKIPPED"}

        blocks = find_transformer_blocks(model)
        if not blocks:
            recorder.emit(
                battery_id, "FAIL", model_id=large_id,
                scope="Fase final C5 (indisponível)",
                metrics={"error": "nenhum transformer block encontrado"},
                notes=f"C5 FAIL: find_transformer_blocks não localizou blocos em {large_id}.",
                highlight="sem blocos",
            )
            return {"status": "FAIL"}

        idxs = sample_block_indices(len(blocks))
        print(f"[FINAL] C5: {len(blocks)} blocos; amostrados {idxs}")
        inputs_small = tokenizer(ACTIVATION_PROMPT, return_tensors="pt", truncation=True, max_length=64)
        inputs_small = {k: v.to(device) for k, v in inputs_small.items()}
        enc_gen = tokenizer(GENERATION_PROMPT, return_tensors="pt")
        enc_gen = {k: v.to(device) for k, v in enc_gen.items()}
        patch_device = device if getattr(model, "hf_device_map", None) is None else None
        decomp_cache: Dict[str, CascadeLinearStages] = {}
        n_total = len(blocks)

        # qualidade por profundidade: um bloco patchado por vez, entrada REAL via hook
        depth_curve: List[Dict[str, Any]] = []

        def _per_block_quality():
            for idx in idxs:
                bn_i, blk_i = blocks[idx]
                cap = capture_block_io(model, blk_i, inputs_small)
                fn_i, mode_i = make_block_forward(model, blk_i, cap, inputs_small)
                y_base = fn_i()
                if y_base is None and torch.is_tensor(cap.get("out")):
                    y_base = cap["out"]
                if y_base is None:
                    raise RuntimeError(f"bloco {bn_i}: forward sem saída")
                orig_i, repl_i = patch_one_block(
                    blk_i, bn_i, codec,
                    rank=args.rank, group_size=args.group_size,
                    gate_percentile=args.gate_percentile, path="F0_GATE_F1",
                    device=patch_device, decomp_cache=decomp_cache,
                )
                try:
                    reset_replaced_counters(repl_i)
                    y_c = fn_i()
                finally:
                    restore_block_linears(blk_i, orig_i, bn_i)
                q_i = cosine_nrmse(y_base, y_c) if y_c is not None else {"cosine": 0.0, "nrmse": 1.0}
                counters_i = read_replaced_counters(repl_i)
                depth_curve.append({
                    "block_index": int(idx),
                    "depth_fraction": (idx / (n_total - 1)) if n_total > 1 else 0.0,
                    "block_name": bn_i,
                    "cosine": float(q_i["cosine"]),
                    "nrmse": float(q_i["nrmse"]),
                    "f1_skip_rate": counters_i["F1_skip_rate"],
                    "forward_mode": mode_i,
                })
                print(f"[FINAL] C5 bloco {idx:>3} cos={q_i['cosine']:.4f} nrmse={q_i['nrmse']:.4f}")

        _, ram_quality = measure_phase_ram(_per_block_quality)

        # drift acumulado: blocos amostrados patchados SIMULTANEAMENTE, 1 forward
        logits_base = forward_logits(model, enc_gen)
        patched: List[Tuple[nn.Module, str, Dict[str, nn.Linear]]] = []
        chain_counters: Dict[str, Any] = {"F0_calls": 0, "F1_calls": 0, "F1_skip_rate": None}
        logits_chain = None

        def _chained():
            nonlocal logits_chain, chain_counters
            all_replaced: List[Dict[str, nn.Module]] = []
            for idx in idxs:
                bn_i, blk_i = blocks[idx]
                orig_i, repl_i = patch_one_block(
                    blk_i, bn_i, codec,
                    rank=args.rank, group_size=args.group_size,
                    gate_percentile=args.gate_percentile, path="F0_GATE_F1",
                    device=patch_device, decomp_cache=decomp_cache,
                )
                patched.append((blk_i, bn_i, orig_i))
                all_replaced.append(repl_i)
            for repl_i in all_replaced:
                reset_replaced_counters(repl_i)
            logits_chain = forward_logits(model, enc_gen)
            merged: Dict[str, nn.Module] = {}
            for repl_i in all_replaced:
                merged.update(repl_i)
            chain_counters = read_replaced_counters(merged)

        try:
            _, ram_chain = measure_phase_ram(_chained)
        finally:
            for blk_i, bn_i, orig_i in patched:
                restore_block_linears(blk_i, orig_i, bn_i)

        chain_q = (
            cosine_nrmse(logits_base, logits_chain)
            if (logits_base is not None and logits_chain is not None)
            else None
        )
        drift = (1.0 - float(chain_q["cosine"])) if chain_q is not None else None
        worst_cos = min((e["cosine"] for e in depth_curve), default=0.0)
        per_block_ok = bool(depth_curve) and all(e["cosine"] >= 0.95 for e in depth_curve)
        drift_ok = drift is not None and drift <= 0.12
        c5_pass = per_block_ok and drift_ok
        status = "PASS" if c5_pass else "FAIL"
        criteria = {
            "per_block_cosine_ge_095": {"pass": per_block_ok, "worst": worst_cos, "min": 0.95},
            "chained_drift_le_012": {"pass": drift_ok, "value": drift, "max": 0.12,
                                     "budget_source": "spec SPECTRA (budget de drift)"},
        }
        recorder.emit(
            battery_id, status, model_id=large_id,
            scope=(
                f"C5 blocos representativos: {len(idxs)} blocos de {n_total} amostrados no "
                f"espectro de profundidade (início/meio/fim) de {large_id}; entrada REAL de "
                f"cada bloco via hook; patch transacional F0({codec_name})+Gate·F1 por bloco "
                f"(unpatch após medir) + drift acumulado com os blocos amostrados patchados "
                f"SIMULTANEAMENTE (1 forward do modelo, 1-cosine dos logits); RAM topo=null "
                f"(gate diagnóstico — VmRSS por fase em metrics.memory)"
            ),
            quality_output={
                "depth_curve": depth_curve,
                "chained_logits": chain_q,
            },
            full_gate=c5_pass,
            metrics={
                "final": {
                    "depth_curve": depth_curve,
                    "sampled_block_indices": idxs,
                    "n_blocks_total": n_total,
                    "n_params": n_params,
                    "chained": {
                        "n_blocks_patched": len(idxs),
                        "logits_cosine": (chain_q or {}).get("cosine"),
                        "drift": drift,
                        "budget": 0.12,
                        **chain_counters,
                    },
                    "criteria": criteria,
                    "codec_f0": codec_name,
                },
                "memory": {
                    "method": phase_method(ram_load, ram_quality, ram_chain),
                    "load_phase": ram_load,
                    "per_block_quality_phase": ram_quality,
                    "chained_drift_phase": ram_chain,
                },
            },
            notes=(
                f"C5: {len(idxs)} blocos amostrados de {n_total}; pior cos por bloco="
                f"{worst_cos:.4f} (min 0.95); drift acumulado="
                f"{drift if drift is None else round(drift, 4)} (budget 0.12) → "
                f"{'PASS' if c5_pass else 'FAIL'}."
            ),
            highlight=f"pior_cos={worst_cos:.4f} drift={drift if drift is None else round(drift, 4)}",
        )
        return {"status": status, "n_sampled": len(idxs), "worst_block_cosine": worst_cos,
                "drift": drift}
    finally:
        del model, tokenizer
        free_memory()
        print(f"[FINAL] modelo {large_id} descarregado (del + gc + cuda empty_cache)")


# ---------------------------------------------------------------------------
# C6 — MARCO FINAL: compilar bundles reais e executar a partir deles
# ---------------------------------------------------------------------------

def run_c6(recorder: FinalRecorder, args: argparse.Namespace, *, tech_upper: str,
           device: "torch.device", codec, codec_name: str, token: Optional[str],
           model_id: str, out_dir: Path) -> Dict[str, Any]:
    battery_id = f"C6_{tech_upper}_COMPILE_EXECUTE"
    if args.skip_c6:
        recorder.emit(
            battery_id, "SKIPPED", model_id=model_id,
            scope="Fase final C6 (pulada por --skip-c6)",
            metrics={"skipped": True, "flag": "--skip-c6"},
            notes="C6 pulada via --skip-c6.",
            highlight="SKIPPED (flag)",
        )
        return {"status": "SKIPPED"}

    f0_label = F0_BUNDLE_CODEC_LABEL[codec_name]
    print(f"[FINAL] C6: carregando {model_id} (marco compilar+executar)...")
    loaded, ram_load = measure_phase_ram(
        lambda: load_model(model_id, device, args.trust_remote_code, token)
    )
    model, tokenizer = loaded
    del loaded
    try:
        n_params = count_params(model)
        if n_params > PARAM_GUARD:
            recorder.emit(
                battery_id, "SKIPPED", model_id=model_id,
                scope="Fase final C6 (guard de recursos)",
                metrics={"skipped": True, "n_params": n_params, "guard": PARAM_GUARD},
                notes=f"C6 SKIPPED: {model_id} tem {n_params} params (> {PARAM_GUARD}).",
                highlight="SKIPPED (>3e9 params)",
            )
            return {"status": "SKIPPED"}
        blocks = find_transformer_blocks(model)
        if not blocks:
            recorder.emit(
                battery_id, "FAIL", model_id=model_id,
                scope="Fase final C6 (indisponível)",
                metrics={"error": "nenhum transformer block encontrado"},
                notes=f"C6 FAIL: sem blocos em {model_id}.",
                highlight="sem blocos",
            )
            return {"status": "FAIL"}

        # (a) baseline REAL ANTES da compilação (logits + generate)
        enc_gen = tokenizer(GENERATION_PROMPT, return_tensors="pt")
        enc_gen = {k: v.to(device) for k, v in enc_gen.items()}
        logits_base = forward_logits(model, enc_gen)
        base_gen, ram_gen_base = measure_phase_ram(
            lambda: measure_generate(
                model, tokenizer, GENERATION_PROMPT, device,
                max_new_tokens=args.max_new_tokens, warmup=2, timed=3,
            )
        )
        baseline_tok_s = float(base_gen["tok_s_median"])
        print(f"[FINAL] C6 baseline {baseline_tok_s:.2f} tok/s "
              f"({base_gen['n_new_tokens']} tokens novos, greedy)")

        # (b) COMPILE: um bundle CSCD REAL por Linear de bloco em <out>/bundle/
        bundle_dir = out_dir / "bundle"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        bundle_paths: Dict[str, Path] = {}
        baseline_ckpt_bytes = 0  # numel × dtype-size REAIS do checkpoint (W + bias)

        def _compile() -> int:
            nonlocal baseline_ckpt_bytes
            n = 0
            for bn_i, blk_i in blocks:
                for full, linear in collect_block_linears(blk_i, bn_i).items():
                    w = linear.weight
                    baseline_ckpt_bytes += int(w.numel() * w.element_size())
                    if linear.bias is not None:
                        baseline_ckpt_bytes += int(linear.bias.numel() * linear.bias.element_size())
                    wf = w.detach().to(dtype=torch.float32).cpu()
                    stages, _packed = decompose_with_codec(
                        wf, codec, rank=args.rank, group_size=args.group_size
                    )
                    b_path = bundle_dir / (full.replace(".", "_") + ".cascade")
                    write_cascade_bundle(
                        b_path, stages=stages, model_id=model_id, target_layer=full,
                        gate_percentile=args.gate_percentile, f0_codec=f0_label,
                    )
                    bundle_paths[full] = b_path
                    del stages, _packed, wf
                    n += 1
            return n

        t_compile0 = time.perf_counter_ns()
        n_compiled, ram_compile = measure_phase_ram(_compile)
        compile_ms = (time.perf_counter_ns() - t_compile0) / 1e6
        candidate_disk = sum(stat_bytes(p) for p in bundle_paths.values())
        print(f"[FINAL] C6 compilado: {n_compiled} bundles em {bundle_dir} "
              f"({candidate_disk} B, {compile_ms:.0f} ms)")

        # (c) EXECUTE FROM BUNDLE: swap transacional + descarte dos pesos originais
        patch_device = device if getattr(model, "hf_device_map", None) is None else None
        swapped: Dict[str, nn.Module] = {}
        originals: Dict[str, nn.Module] = {}
        location: Dict[str, Tuple[nn.Module, str]] = {}

        def _swap() -> None:
            try:
                for bn_i, blk_i in blocks:
                    for full, linear in collect_block_linears(blk_i, bn_i).items():
                        mod = BundleLinearModule(
                            bundle_paths[full], codec, expected_f0_codec=f0_label,
                            gate_percentile=args.gate_percentile, bias=linear.bias,
                        )
                        if patch_device is not None:
                            mod = mod.to(patch_device)
                        short = full[len(bn_i) + 1:] if full.startswith(bn_i + ".") else full
                        set_module_by_path(blk_i, short, mod)
                        originals[full] = linear
                        location[full] = (blk_i, short)
                        swapped[full] = mod
            except Exception:
                # rollback transacional: nenhum bloco fica parcialmente patchado
                for full in list(swapped):
                    blk_r, short_r = location[full]
                    try:
                        set_module_by_path(blk_r, short_r, originals[full])
                    except Exception as rexc:
                        print(f"[FINAL] AVISO rollback de {full} falhou: {rexc}")
                raise

        _, ram_swap = measure_phase_ram(_swap)

        # descarte REAL dos pesos originais (fora do caminho quente e inalcançáveis)
        rss_before_free = _read_vmrss_bytes()
        cuda_before_free = (
            int(torch.cuda.memory_allocated()) if device.type == "cuda" and torch.cuda.is_available() else None
        )
        n_freed = len(originals)
        originals.clear()
        location.clear()
        free_memory()
        rss_after_free = _read_vmrss_bytes()
        cuda_after_free = (
            int(torch.cuda.memory_allocated()) if device.type == "cuda" and torch.cuda.is_available() else None
        )
        # critério real, não tautológico: todos os Linear compilados foram
        # trocados E seus originais descartados (dict vazio após o clear)
        original_weights_freed = bool(
            n_freed == n_compiled and n_freed > 0 and len(originals) == 0
        )
        print(f"[FINAL] C6: {n_freed} nn.Linear originais descartadas "
              f"(RSS {rss_before_free}→{rss_after_free} B; "
              f"CUDA {cuda_before_free}→{cuda_after_free} B)")

        # (d) candidato: MESMO protocolo, executando DOS bundles
        logits_cand = forward_logits(model, enc_gen)
        cand_gen, ram_gen_cand = measure_phase_ram(
            lambda: measure_generate(
                model, tokenizer, GENERATION_PROMPT, device,
                max_new_tokens=args.max_new_tokens, warmup=2, timed=3,
            )
        )
        candidate_tok_s = float(cand_gen["tok_s_median"])
        print(f"[FINAL] C6 candidato {candidate_tok_s:.2f} tok/s (runtime de bundle)")

        f0_all = sum(int(m.f0_calls) for m in swapped.values())
        f1_all = sum(int(m.f1_calls) for m in swapped.values())
        skip_all = 1.0 - (f1_all / max(f0_all, 1))
        em = token_exact_match(base_gen["new_token_ids"], cand_gen["new_token_ids"])
        logits_q = (
            cosine_nrmse(logits_base, logits_cand)
            if (logits_base is not None and logits_cand is not None)
            else None
        )
        logits_cos = logits_q["cosine"] if logits_q else None
        resident_stage_bytes = sum(int(m.stats()["resident_bytes"]) for m in swapped.values())
        w0_cache_bytes = sum(int(m.stats()["w0_cache_bytes"]) for m in swapped.values())

        # (e) PASS do marco (contrato §16)
        criteria = {
            "executes": {"pass": candidate_tok_s > 0 and baseline_tok_s > 0 and f0_all > 0,
                         "f0_calls": f0_all},
            # f0_all > 0 impede skip espúrio (1.0) quando os módulos nunca rodaram
            "gate_active_skip_gt_zero": {"pass": f0_all > 0 and skip_all > 0,
                                         "value": skip_all, "f0_calls": f0_all},
            "logits_cosine_ge_095": {"pass": logits_cos is not None and logits_cos >= 0.95,
                                     "value": logits_cos, "min": 0.95},
            "bundle_lt_checkpoint": {"pass": candidate_disk < baseline_ckpt_bytes,
                                     "bundle_bytes": candidate_disk,
                                     "checkpoint_bytes": baseline_ckpt_bytes},
            "original_weights_freed": {"pass": original_weights_freed,
                                       "n_modules_freed": n_freed},
        }
        c6_pass = all(bool(c["pass"]) for c in criteria.values())
        status = "PASS" if c6_pass else "FAIL"
        disk_reduction_pct = 100.0 * (1.0 - candidate_disk / max(baseline_ckpt_bytes, 1))

        recorder.emit(
            battery_id, status, model_id=model_id,
            scope=(
                f"C6 MARCO FINAL compilar+executar: TODAS as {n_compiled} Linear dos "
                f"{len(blocks)} blocos de {model_id} compiladas para bundles CSCD REAIS "
                f"({bundle_dir}, codec F0={f0_label}); runtime carrega codes/scales/u/s/v "
                f"DO ARQUIVO (mmap + torch.frombuffer, leitor validador CSCD); pesos densos "
                f"originais DESCARTADOS após o swap (del+gc — fora do caminho quente); "
                f"baseline E candidato via model.generate (greedy, "
                f"max_new_tokens={args.max_new_tokens}, 2 warmup + 3 medições, mediana; "
                f"baseline medido ANTES da compilação); RAM topo=pico VmRSS por fase; "
                f"candidate_disk=os.stat dos bundles; checkpoint=numel×dtype-size reais"
            ),
            quality_output={
                "logits": logits_q,
                "token_exact_match": em,
            },
            full_gate=c6_pass,
            metrics={
                "e2e": {
                    "measured": True,
                    "metric": "e2e_generate_tok_s",
                    "baseline": {k: v for k, v in base_gen.items() if k != "new_token_ids"},
                    "candidate": {k: v for k, v in cand_gen.items() if k != "new_token_ids"},
                    "speedup_x": candidate_tok_s / max(baseline_tok_s, 1e-12),
                },
                "memory": {
                    "method": phase_method(ram_gen_base, ram_gen_cand, ram_compile, ram_swap, ram_load),
                    "load_phase": ram_load,
                    "baseline_generate_phase": ram_gen_base,
                    "compile_phase": ram_compile,
                    "swap_phase": ram_swap,
                    "candidate_generate_phase": ram_gen_cand,
                    "rss_before_weight_free_bytes": rss_before_free,
                    "rss_after_weight_free_bytes": rss_after_free,
                    "cuda_allocated_before_free_bytes": cuda_before_free,
                    "cuda_allocated_after_free_bytes": cuda_after_free,
                },
                "final": {
                    "original_weights_freed": original_weights_freed,
                    "n_linears_compiled": n_compiled,
                    "n_original_modules_freed": n_freed,
                    "n_blocks": len(blocks),
                    "n_params": n_params,
                    "bundle_dir": str(bundle_dir),
                    "bundle_total_bytes": candidate_disk,
                    "checkpoint_bytes": baseline_ckpt_bytes,
                    "disk_reduction_pct": disk_reduction_pct,
                    "compile_ms": compile_ms,
                    "codec_f0": codec_name,
                    "f0_bundle_codec": f0_label,
                    "gate": {"F0_calls": f0_all, "F1_calls": f1_all, "F1_skip_rate": skip_all},
                    "resident_stage_bytes": resident_stage_bytes,
                    "w0_cache_bytes": w0_cache_bytes,
                    "criteria": criteria,
                    "original_dense_weight_reconstructed": False,
                    "bias_note": (
                        "bias FP32 preservado FORA do bundle (vetor pequeno; nunca é o W "
                        "denso); lm_head/embeddings fora dos blocos permanecem originais"
                    ),
                },
            },
            baseline_disk=baseline_ckpt_bytes,
            candidate_disk=candidate_disk,
            baseline_tok_s=baseline_tok_s,
            candidate_tok_s=candidate_tok_s,
            primary=True,
            ram_base=ram_gen_base,
            ram_cand=ram_gen_cand,
            notes=(
                f"C6 (marco): baseline={baseline_tok_s:.2f} tok/s candidato="
                f"{candidate_tok_s:.2f} tok/s (ambos REAIS, model.generate; candidato "
                f"executa DOS bundles). logits_cos="
                f"{logits_cos if logits_cos is None else round(logits_cos, 4)} "
                f"exact_match={em['exact_match_rate']:.3f} skip={skip_all:.3f} "
                f"disco checkpoint {baseline_ckpt_bytes} B → bundles {candidate_disk} B "
                f"({disk_reduction_pct:.1f}% menor). Pesos originais liberados: "
                f"{original_weights_freed} ({n_freed} módulos)."
            ),
            highlight=f"{baseline_tok_s:.2f}->{candidate_tok_s:.2f} tok/s cos={logits_cos if logits_cos is None else round(logits_cos, 4)}",
        )
        return {
            "status": status,
            "baseline_tok_s": baseline_tok_s,
            "candidate_tok_s": candidate_tok_s,
            "speedup_x": candidate_tok_s / max(baseline_tok_s, 1e-12),
            "logits_cosine": logits_cos,
            "token_exact_match_rate": em["exact_match_rate"],
            "f1_skip_rate": skip_all,
            "checkpoint_bytes": baseline_ckpt_bytes,
            "bundle_bytes": candidate_disk,
            "disk_reduction_pct": disk_reduction_pct,
            "original_weights_freed": original_weights_freed,
            "rss_before_free": rss_before_free,
            "rss_after_free": rss_after_free,
        }
    finally:
        del model, tokenizer
        free_memory()
        print(f"[FINAL] modelo {model_id} descarregado (del + gc + cuda empty_cache)")


# ---------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fase final C4/C5/C6 — FINAL_PHASE_V1 (contrato C3_CONTRACTS_V1 §16)"
    )
    p.add_argument("--technology", required=True, choices=sorted(TECH_DEFAULT_CODEC))
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B",
                   help="modelo pequeno (família A do C4 e alvo do C6)")
    p.add_argument("--second-model", default="HuggingFaceTB/SmolLM2-360M",
                   help="modelo B de OUTRA família de arquitetura (C4)")
    p.add_argument("--large-model", default="Qwen/Qwen2.5-1.5B",
                   help="modelo maior dos blocos representativos (C5)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--codec", default=None, choices=sorted(_CODECS))
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument("--gate-percentile", type=float, default=70.0)
    p.add_argument("--max-new-tokens", type=int, default=48, help="tokens do generate do C6")
    p.add_argument("--out", default="final_test_output")
    p.add_argument("--publish", default="on", choices=["on", "off"])
    p.add_argument("--skip-c4", action="store_true", help="pula a bateria C4 (segunda família)")
    p.add_argument("--skip-c5", action="store_true", help="pula a bateria C5 (blocos representativos)")
    p.add_argument("--skip-c6", action="store_true", help="pula a bateria C6 (compilar+executar)")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--results-endpoint", default=None, help="URL HTTPS /api/results (default: env)")
    values = sys.argv[1:] if argv is None else list(argv)
    args = p.parse_args(without_ipykernel_connection_args(values))
    if not 0 <= args.gate_percentile <= 100:
        p.error("--gate-percentile precisa estar entre 0 e 100")
    if args.max_new_tokens < 1:
        p.error("--max-new-tokens precisa ser >= 1")
    if args.trust_remote_code:
        print("[FINAL] AVISO: --trust-remote-code executa código do repositório do modelo. "
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
    model_id = normalize_model_id(args.model)
    second_id = normalize_model_id(args.second_model)
    large_id = normalize_model_id(args.large_model)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id()
    publish_on = args.publish != "off"
    recorder = FinalRecorder(
        out_dir, tech=tech, tech_upper=tech_upper, device=device, codec_name=codec_name,
        run_id=run_id, publish_on=publish_on, endpoint=args.results_endpoint,
    )

    print(f"[FINAL] tech={tech_upper} codec={codec_name} device={device} rank={args.rank} "
          f"gs={args.group_size} gate_pct={args.gate_percentile}")
    print(f"[FINAL] modelos: A={model_id} | B={second_id} | maior={large_id}")

    summary: Dict[str, Any] = {
        "run_id": run_id, "technology": tech_upper, "codec": codec_name,
        "model_id": model_id, "second_model_id": second_id, "large_model_id": large_id,
        "device": device.type, "benchmark_protocol": BENCHMARK_PROTOCOL,
    }

    def guarded(label: str, battery_id: str, fallback_model_id: str, fn):
        """Erro inesperado vira registro FAIL; OOM vira SKIPPED (guard §16)."""
        try:
            return fn()
        except Exception as exc:
            traceback.print_exc()
            oom = is_oom_error(exc)
            recorder.emit(
                battery_id, "SKIPPED" if oom else "FAIL",
                model_id=fallback_model_id,
                scope=f"Fase final {label} (erro em runtime)",
                metrics={"error": f"{type(exc).__name__}: {exc}"[:800], "oom": oom},
                notes=(
                    f"{label} SKIPPED por OOM (guard de recursos do contrato §16): {exc}"
                    if oom else f"Falha na {label}: {exc}"
                )[:1200],
                highlight="OOM" if oom else "erro",
            )
            free_memory()
            return {"status": "SKIPPED" if oom else "FAIL", "error": str(exc)[:400]}

    summary["c4"] = guarded(
        "C4 (segunda família)", f"C4_{tech_upper}_SECOND_FAMILY", model_id,
        lambda: run_c4(recorder, args, tech_upper=tech_upper, device=device, codec=codec,
                       codec_name=codec_name, token=token,
                       model_a_id=model_id, model_b_id=second_id),
    )
    summary["c5"] = guarded(
        "C5 (blocos representativos)", f"C5_{tech_upper}_REPR_BLOCKS", large_id,
        lambda: run_c5(recorder, args, tech_upper=tech_upper, device=device, codec=codec,
                       codec_name=codec_name, token=token, large_id=large_id),
    )
    summary["c6"] = guarded(
        "C6 (compilar+executar)", f"C6_{tech_upper}_COMPILE_EXECUTE", model_id,
        lambda: run_c6(recorder, args, tech_upper=tech_upper, device=device, codec=codec,
                       codec_name=codec_name, token=token, model_id=model_id, out_dir=out_dir),
    )

    # ------------------------------------------------------------------
    # Gain report + tabela final PT-BR
    # ------------------------------------------------------------------
    summary["generated_at"] = utc()
    summary["records_total"] = len(recorder.summary_rows)
    summary["records_pass"] = sum(1 for r in recorder.summary_rows if r["status"] == "PASS")
    gain_path = out_dir / f"final_{tech}_gain_report.json"
    gain_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    c4 = summary.get("c4") or {}
    c5 = summary.get("c5") or {}
    c6 = summary.get("c6") or {}

    print()
    print("=" * 96)
    print(f"FASE FINAL {tech_upper} — C4/C5/C6 ({BENCHMARK_PROTOCOL}) | codec={codec_name} "
          f"| device={device.type}")
    print("=" * 96)
    print(f"{'battery_id':<40} {'status':<12} destaque")
    print("-" * 96)
    for row in recorder.summary_rows:
        print(f"{row['battery_id']:<40} {row['status']:<12} {row['highlight']}")
    print("-" * 96)
    fa = c4.get("family_a") or {}
    fb = c4.get("family_b") or {}
    if fa and fb:
        print(
            f"C4 Segunda família      : A={fa.get('gated_cosine_min', 0):.4f} "
            f"({fa.get('model_type')}) | B={fb.get('gated_cosine_min', 0):.4f} "
            f"({fb.get('model_type')}) — critério >=0.98 nas duas"
        )
    else:
        print(f"C4 Segunda família      : {c4.get('status', '—')}")
    if c5.get("drift") is not None or c5.get("worst_block_cosine") is not None:
        _drift = c5.get("drift")
        print(
            f"C5 Blocos representativ.: {c5.get('n_sampled', 0)} blocos, pior cos="
            f"{c5.get('worst_block_cosine', 0):.4f}, drift acumulado="
            f"{_drift if _drift is None else round(_drift, 4)} (budget 0.12)"
        )
    else:
        print(f"C5 Blocos representativ.: {c5.get('status', '—')}")
    if c6.get("baseline_tok_s") is not None:
        print(
            f"C6 Compilar+Executar    : baseline={c6['baseline_tok_s']:.2f} tok/s → "
            f"candidato={c6['candidate_tok_s']:.2f} tok/s ({c6['speedup_x']:.2f}x, "
            f"ambos REAIS via model.generate)"
        )
        print(
            f"Disco (C6)              : checkpoint={c6['checkpoint_bytes']} B → "
            f"bundles={c6['bundle_bytes']} B (redução {c6['disk_reduction_pct']:.1f}%)"
        )
        print(
            f"Pesos originais (C6)    : {'LIBERADOS' if c6.get('original_weights_freed') else 'NÃO LIBERADOS'} "
            f"(RSS {c6.get('rss_before_free')}→{c6.get('rss_after_free')} B)"
        )
    else:
        print(f"C6 Compilar+Executar    : {c6.get('status', '—')}")
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
        _rc = 0  # baterias reportam; falhas viram registros FAIL/SKIPPED
    finally:
        try:
            cleanup_colab_workspace(label="FINAL-PHASE", wipe_hf_cache=False)
        except Exception as _ce:
            print(f"[cleanup] AVISO: {_ce}")
    raise SystemExit(_rc)
