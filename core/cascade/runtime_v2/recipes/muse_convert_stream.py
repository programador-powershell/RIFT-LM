#!/usr/bin/env python3
"""Conversao REAL da Muse-Glimmer-30B (LM completo) por STREAMING remoto.

Nao ha disco para os 55.7 GB de fonte: cada tensor e lido por HTTP Range em
chunks de linhas direto dos shards no HF, codificado e gravado — pico de
disco = so o bundle (~16 GB); pico de RAM = chunk + planos.

Receita v2.1 (ditada pela bateria E2E do 0.5B):
  - corpo 2D:        q4k/g32 + clip  (piso g32: degrau g64 custou +1.44 PPL)
  - v_proj:          q8r (rowwise int8)  — promocao barata (+0.04 GB no 30B)
  - lm_head:         q8r                 — cabeca 4.5 bpw custou +1.09 PPL
  - embed_tokens:    q4k/g32 (lookup; nao entra em GEMV critico)
  - norms/1D:        raw bf16
  - runtime:         ativacoes int8 por grupo-32 (kernels v2.1)
  - AWQ-lite:        NAO aplicado (exige ativacoes da Muse -> passo Colab);
                     upside medido no proxy 0.5B: ~-0.9 PPL
Sem escada: 1 degrau por papel = 1 passe de rede por tensor. Gate registrado
por tensor; quality_flag quando cos < 0.995 (sem resgate — reportar, nao
mascarar).
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/root/muse/runtime_fixed")
from cascade_runtime_v2.codec import encode_qk
from cascade_runtime_v2.q4k_pack import SUP, pack_q4k

REPO = "meta-models/Muse-Glimmer-30B"
OUT = Path("/root/muse/muse-30b-cascade-v21")
CHUNK_ROWS = 4096
CLIPS = (1.0, 0.975, 0.95, 0.925, 0.9)
GATE_COS = 0.995
KV_RESERVE_GIB = 1.5


def build_tensor_map(fs):
    tmap = {}
    for shard in sorted(fs.glob(f"{REPO}/model-*.safetensors")):
        with fs.open(shard, "rb") as fh:
            (hlen,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(hlen))
        for n, m in header.items():
            if n != "__metadata__":
                tmap[n] = (shard, 8 + hlen, m)
    return tmap


def role_of(name: str) -> str:
    if name == "lm_head.weight":
        return "q8r"
    if "v_proj" in name:
        return "q8r"
    if "embed_tokens" in name:
        return "q4k32"
    return "q4k32"


def stream_rows(fs, shard, base, meta, r0, r1):
    rows, cols = meta["shape"]
    start = meta["data_offsets"][0]
    with fs.open(shard, "rb", block_size=64 * 1024 * 1024) as fh:
        fh.seek(base + start + r0 * cols * 2)
        raw = fh.read((r1 - r0) * cols * 2)
    return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16) \
        .reshape(r1 - r0, cols).to(torch.float32)


def convert_2d(fs, name, shard, base, meta, kind, blob_path):
    rows, cols = meta["shape"]
    dot = nw = nd = se = 0.0
    absmax = 0.0
    written = 0
    h = hashlib.sha256()
    with open(blob_path, "wb") as out:
        for r0 in range(0, rows, CHUNK_ROWS):
            r1 = min(r0 + CHUNK_ROWS, rows)
            w = stream_rows(fs, shard, base, meta, r0, r1)
            absmax = max(absmax, float(w.abs().max().item()))
            if kind == "q4k32":
                pad = (-cols) % SUP
                wp = torch.nn.functional.pad(w, (0, pad)) if pad else w
                dq, planes = encode_qk(wp, g=32, bits=4, clips=CLIPS)
                blob = pack_q4k(planes).tobytes()
                dq = dq[:, :cols]
            else:  # q8r
                s = w.abs().amax(1, keepdim=True).clamp_min(1e-12) / 127.0
                w8 = (w / s).round().clamp(-127, 127)
                dq = w8 * s
                blob = w8.to(torch.int8).numpy().tobytes() + \
                    s.squeeze(1).to(torch.float16).numpy().tobytes()
            out.write(blob)
            h.update(blob)
            written += len(blob)
            dot += (w * dq).sum(dtype=torch.float64).item()
            nw += w.pow(2).sum(dtype=torch.float64).item()
            nd += dq.pow(2).sum(dtype=torch.float64).item()
            se += (dq - w).pow(2).sum(dtype=torch.float64).item()
            del w, dq
    cos = dot / max(math.sqrt(nw * nd), 1e-300)
    nrmse = math.sqrt(se / (rows * cols)) / max(absmax, 1e-300)
    return {"cosine": round(cos, 6), "nrmse": round(nrmse, 6),
            "output_bytes": written, "sha256": h.hexdigest()[:16]}


def main() -> int:
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    (OUT / "tensors").mkdir(parents=True, exist_ok=True)
    tmap = build_tensor_map(fs)
    wanted = [n for n in sorted(tmap) if n.startswith("model.language_model.")
              or n == "lm_head.weight"]
    manifest = {"format": "CASCADE-Q4K/2.1", "model_id": REPO + " (LM apenas)",
                "receita": "corpo q4k/g32+clip | v_proj+lm_head q8r | "
                           "emb q4k/g32 | acts int8/g32 | AWQ pendente (Colab)",
                "tensors": []}
    t0 = time.time()
    total_out = total_src = 0
    below = []
    peak = [0]

    def rss():
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                v = int(line.split()[1]) * 1024
                peak[0] = max(peak[0], v)
                return v / 2**30
        return 0.0

    done_names = set()
    man_path = OUT / "cascade_manifest.json"
    if man_path.exists():
        manifest = json.loads(man_path.read_text())
        done_names = {t["name"] for t in manifest["tensors"]}
        total_out = sum(t["output_bytes"] for t in manifest["tensors"])
        total_src = sum(t.get("source_bytes", 0) for t in manifest["tensors"])
        below = [t["name"] for t in manifest["tensors"] if t.get("quality_flag")]
        print(f"[resume] {len(done_names)} tensores ja convertidos", flush=True)

    for i, name in enumerate(wanted, 1):
        if name in done_names:
            continue
        shard, base, meta = tmap[name]
        shape = meta["shape"]
        src_bytes = int(np.prod(shape)) * 2
        slug = name.replace("/", "_")
        if len(shape) != 2:
            w = stream_rows(fs, shard, base, {**meta, "shape": [1, shape[0]]},
                            0, 1).reshape(shape)
            blob = OUT / "tensors" / f"{slug}.raw"
            data = w.to(torch.bfloat16).view(torch.uint8).numpy().tobytes()
            blob.write_bytes(data)
            rec = {"name": name, "shape": shape, "rung": "raw_1d",
                   "source_bytes": src_bytes, "output_bytes": len(data),
                   "blob": blob.name}
        else:
            kind = role_of(name)
            blob = OUT / "tensors" / f"{slug}.{kind}"
            m = convert_2d(fs, name, shard, base, meta, kind, blob)
            rec = {"name": name, "shape": shape, "rung": kind,
                   "source_bytes": src_bytes, "blob": blob.name, **m,
                   "bpw": round(m["output_bytes"] * 8 / int(np.prod(shape)), 3)}
            if kind == "q4k32" and m["cosine"] < GATE_COS:
                rec["quality_flag"] = "abaixo_do_gate_verificar_e2e"
                below.append(name)
        manifest["tensors"].append(rec)
        total_out += rec["output_bytes"]
        total_src += src_bytes
        man_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
        print(f"[{i:>4}/{len(wanted)}] {name} {tuple(shape)} -> "
              f"{rec['rung']} cos={rec.get('cosine', 1.0)} "
              f"({rec['output_bytes'] / 1e6:.1f} MB) rss={rss():.2f} GiB "
              f"| bundle {total_out / 1e9:.2f} GB", flush=True)

    need_gib = total_out / 2**30 + KV_RESERVE_GIB
    manifest["summary"] = {
        "all_tensors_passed_gate": not below,
        "below_gate_tensor_count": len(below),
        "below_gate_tensors": below,
        "source_bytes": total_src, "bundle_bytes": total_out,
        "reducao_pct": round(100 * (1 - total_out / total_src), 2),
        "bpw_medio": round(total_out * 8 / (total_src / 2), 3),
        "conversion_seconds": round(time.time() - t0, 1),
        "peak_rss_gib": round(peak[0] / 2**30, 3),
        "residencia": {
            "necessario_com_reserva_gib": round(need_gib, 2),
            "classes": {f"maquina_{t}gb": {
                "orcamento_gib": float(max(t - 8, t / 2)),
                "cabe": need_gib <= max(t - 8, t / 2),
                "folga_gib": round(max(t - 8, t / 2) - need_gib, 2)}
                for t in (16, 24, 32, 48)},
        },
    }
    man_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    if below:
        print(f"\nATENCAO: {len(below)} tensores abaixo do gate: {below[:8]}")
    s = manifest["summary"]
    print(f"\nMUSE-CONVERSAO-COMPLETA: {total_src / 1e9:.2f} GB -> "
          f"{total_out / 1e9:.2f} GB ({s['reducao_pct']}%, {s['bpw_medio']} "
          f"bpw) em {s['conversion_seconds'] / 60:.0f} min | RSS pico "
          f"{s['peak_rss_gib']} GiB")
    print(json.dumps(s["residencia"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
