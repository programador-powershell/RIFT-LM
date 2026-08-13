"""cascade_convert_v2 — conversor otimizado para o executor cascade_runtime_v2.

Otimizacoes sobre o conversor v1 (auditoria + fisica medida):
  1. SAIDA NO FORMATO DO EXECUTOR: blobs q4k de 144 B/super-bloco prontos para
     mmap + GEMV fundido — zero repack na carga (v1 gravava int4 proprio que o
     executor tinha que dequantar para FP32).
  2. Escada nova (BF16/F16/F32): q4k/g64+clip -> q4k/g32+clip. SEM fallback raw
     para matrizes 2D: o pior caso e 4.5 bpw (v1: 16 bpw). Os degraus q5k/q6k
     estao PLANEJADOS e nao implementados — dependem de kernel 5/6-bit; LADDER
     abaixo e a fonte da verdade.
  3. Embeddings/lm_head: q4k/g32 direto com cosseno MEDIDO no manifest (na
     Muse-Glimmer: 0.9969 nos dois) em vez de raw 16 bpw do v1 — economiza
     ~4.0 GB no 30B. O degrau q6k para embeddings tambem e pendencia de kernel.

  ATENCAO — MUDANCA DE POLITICA: sem fallback raw, quando nenhum degrau passa o
  gate o ultimo degrau e gravado ASSIM MESMO (RESCUE_LAST_RUNG) e o tensor sai
  marcado com quality_flag="abaixo_do_gate_verificar_e2e". Isso ROMPE o
  invariante do conversor v1 (contrato §29.5: "quando nada passa, o resultado
  continua sendo passthrough exato"). O trade e deliberado — troca 16 bpw exatos
  por 4.5 bpw com perda declarada — mas todo consumidor do bundle precisa olhar
  quality_flag antes de tratar o tensor como aprovado.
  4. Exclusao padrao corrigida: cobre token_embd/output.weight (nomenclatura
     GGUF) alem de embed/lm_head (HF) — gap da auditoria v1.
  5. RSS limitado por chunking de LINHAS no encode (q4k e row-independent):
     nem o embedding de 2.69 GB materializa fp32 inteiro.
  6. Relatorio de residencia por CLASSE DE MAQUINA (8/16/24/40 GiB), regra
     orcamento = RAM total - 8 GiB.
Fonte suportada: diretorio HF com *.safetensors (BF16/F16/F32). Fontes GGUF
ja quantizadas continuam no v1 (recomprimir GGUF e beco sem saida medido).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .codec import encode_qk, kq_bpw
from .q4k_pack import SUP, pack_q4k

GATE_COS = 0.995
GATE_NRMSE = 0.05
LADDER = (("q4k", 64, 4), ("q4k", 32, 4))
CLIPS = (1.0, 0.975, 0.95, 0.925, 0.9)
EXCLUDE_TOKENS = ("embed_tokens", "embedding", "embeddings", "lm_head",
                  "token_embd", "output.weight")
# RAM TOTAL da máquina (não o orçamento). O orçamento sai daqui pela regra do
# contrato §28: total - 8 GiB, com piso de metade. Antes esta tupla guardava o
# orçamento E o relatório subtraía 8 de novo, então a linha rotulada
# "maquina_24gb" recebia orçamento de 8 GiB e imprimia NÃO CABE para um bundle
# de 15,54 GiB que CABE em 24 GB — subtração dupla.
MACHINE_TOTAL_GIB = (16, 24, 32, 48)
KV_RUNTIME_RESERVE_GIB = 1.5
CHUNK_ROWS = 4096


def _is_excluded(name: str) -> bool:
    low = name.lower()
    return any(t in low for t in EXCLUDE_TOKENS)


def _f64_metrics(w: torch.Tensor, dq: torch.Tensor, absmax: float) -> dict:
    dot = (w * dq).sum(dtype=torch.float64).item()
    nw = w.pow(2).sum(dtype=torch.float64).item()
    nd = dq.pow(2).sum(dtype=torch.float64).item()
    se = (dq - w).pow(2).sum(dtype=torch.float64).item()
    cos = dot / max(math.sqrt(nw * nd), 1e-300)
    nrmse = math.sqrt(se / w.numel()) / max(absmax, 1e-300)
    return {"cosine": round(cos, 6), "nrmse": round(nrmse, 6)}


def _encode_chunked(w: torch.Tensor, g: int, bits: int) -> tuple[np.ndarray, dict]:
    """Codifica por blocos de linhas (RSS limitado) e mede qualidade em f64."""
    rows, cols = int(w.shape[0]), int(w.shape[1])
    pad = (-cols) % SUP
    absmax = float(w.abs().max().item())
    blobs = []
    dot = nw = nd = se = 0.0
    for r0 in range(0, rows, CHUNK_ROWS):
        wc = w[r0:r0 + CHUNK_ROWS].to(torch.float32)
        if pad:
            wc = torch.nn.functional.pad(wc, (0, pad))
        dq, planes = encode_qk(wc, g=g, bits=bits, clips=CLIPS)
        blobs.append(pack_q4k(planes))
        wc_o = wc[:, :cols]
        dq_o = dq[:, :cols]
        dot += (wc_o * dq_o).sum(dtype=torch.float64).item()
        nw += wc_o.pow(2).sum(dtype=torch.float64).item()
        nd += dq_o.pow(2).sum(dtype=torch.float64).item()
        se += (dq_o - wc_o).pow(2).sum(dtype=torch.float64).item()
        del wc, dq, planes, wc_o, dq_o
    packed = np.vstack(blobs)
    cos = dot / max(math.sqrt(nw * nd), 1e-300)
    nrmse = math.sqrt(se / (rows * cols)) / max(absmax, 1e-300)
    return packed, {"cosine": round(cos, 6), "nrmse": round(nrmse, 6)}


def convert_tensor(name: str, w: torch.Tensor, out_dir: Path) -> dict:
    rows, cols = int(w.shape[0]), int(w.shape[1])
    rec: dict = {"name": name, "shape": [rows, cols], "source_dtype": "BF16",
                 "source_bytes": rows * cols * 2, "ladder": []}
    excluded = _is_excluded(name)
    rungs = (("q4k", 32, 4),) if excluded else LADDER
    if excluded:
        rec["selection"] = ("embedding_like -> q4k/g32 direto com cosseno "
                            "MEDIDO no manifest (v1 mandava raw 16 bpw); "
                            "degrau q6k e pendencia de kernel 6-bit")
    for codec, g, bits in rungs:
        t0 = time.time()
        packed, m = _encode_chunked(w, g, bits)
        attempt = {"rung": f"{codec}/g{g}", "bpw": round(kq_bpw(bits, g), 4),
                   **m, "s": round(time.time() - t0, 1)}
        rec["ladder"].append(attempt)
        last = (codec, g, bits) == rungs[-1]
        if (m["cosine"] >= GATE_COS and m["nrmse"] <= GATE_NRMSE) or last:
            attempt["selected"] = True
            if last and m["cosine"] < GATE_COS:
                attempt["gate"] = "RESCUE_LAST_RUNG"
                rec["quality_flag"] = "abaixo_do_gate_verificar_e2e"
            blob = out_dir / f"{name.replace('/', '_')}.q4k"
            blob.write_bytes(packed.tobytes())
            rec.update({
                "rung": attempt["rung"], "bits": bits, "group": g,
                "output_bytes": int(packed.nbytes),
                "bpw_real": round(packed.nbytes * 8 / (rows * cols), 4),
                "cosine": m["cosine"], "nrmse": m["nrmse"],
                "blob": blob.name,
                "sha256": hashlib.sha256(packed.tobytes()).hexdigest()[:16],
                "cols_padded": cols + ((-cols) % SUP),
            })
            return rec
        del packed
    return rec


def residency_report(total_resident_bytes: int) -> dict:
    """Veredito por classe de máquina, regra do §28: orçamento = total - 8 GiB
    (piso metade). `folga_gib` acompanha o veredito porque a margem importa: um
    bundle projetado que passa por 0,4 GiB não passa se a projeção errar 3%."""
    need_gib = total_resident_bytes / 2**30 + KV_RUNTIME_RESERVE_GIB
    ladder = {}
    for total in MACHINE_TOTAL_GIB:
        budget = max(total - 8, total / 2)
        fits = need_gib <= budget
        ladder[f"maquina_{total:.0f}gb_orcamento_{budget:.0f}gib"] = {
            "veredito": "CABE" if fits else "NAO CABE",
            "folga_gib": round(budget - need_gib, 2),
        }
    return {"resident_gib": round(total_resident_bytes / 2**30, 2),
            "necessario_com_reserva_gib": round(need_gib, 2),
            "kv_runtime_reserve_gib": KV_RUNTIME_RESERVE_GIB,
            "regra_orcamento": "total - 8 GiB, piso total/2 (contrato §28)",
            "classes": ladder}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model-id", default="")
    args = ap.parse_args()

    from safetensors import safe_open
    inp = Path(args.input)
    out = Path(args.output)
    (out / "tensors").mkdir(parents=True, exist_ok=True)
    files = sorted(inp.glob("*.safetensors"))
    if not files:
        raise SystemExit(f"nenhum .safetensors em {inp}")

    manifest = {"format": "CASCADE-Q4K/2.0", "model_id": args.model_id,
                "executor": "cascade_runtime_v2 (q4k_gemv_i8, 144B/super)",
                "gate": {"cosine_min": GATE_COS, "nrmse_max": GATE_NRMSE},
                "tensors": []}
    t0 = time.time()
    total_out = total_src = 0
    peak = [0]

    def rss():
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                v = int(line.split()[1]) * 1024
                peak[0] = max(peak[0], v)
                return v
        return 0

    for f in files:
        with safe_open(str(f), framework="pt") as sf:
            for name in sorted(sf.keys()):
                w = sf.get_tensor(name)
                if w.ndim != 2 or w.numel() < 4096:
                    raw = out / "tensors" / f"{name.replace('/', '_')}.raw"
                    raw.write_bytes(w.contiguous().view(torch.uint8).numpy()
                                    .tobytes() if w.dtype == torch.bfloat16
                                    else w.numpy().tobytes())
                    manifest["tensors"].append(
                        {"name": name, "shape": list(w.shape),
                         "rung": "raw_1d", "output_bytes": raw.stat().st_size,
                         "blob": raw.name})
                    total_out += raw.stat().st_size
                    total_src += w.numel() * w.element_size()
                    continue
                rec = convert_tensor(name, w.to(torch.float32), out / "tensors")
                manifest["tensors"].append(rec)
                total_out += rec["output_bytes"]
                total_src += rec["source_bytes"]
                print(f"[{len(manifest['tensors']):>4}] {name} "
                      f"{tuple(w.shape)} -> {rec['rung']} "
                      f"cos={rec.get('cosine', 1.0)} "
                      f"({rec['output_bytes'] / 1e6:.1f} MB) rss={rss() / 2**30:.2f} GiB",
                      flush=True)
                del w

    # Sem fallback raw, um tensor pode sair ABAIXO do gate. Isso tem de aparecer
    # no resumo: quem consome o bundle não vai varrer 700 tensores procurando
    # quality_flag, e um bundle com tensor reprovado não é um bundle aprovado.
    below = [t["name"] for t in manifest["tensors"] if t.get("quality_flag")]
    manifest["summary"] = {
        "source_bytes": total_src, "bundle_bytes": total_out,
        "reducao_pct": round(100 * (1 - total_out / total_src), 2),
        "bpw_medio": round(total_out * 8 / (total_src / 2), 3),
        "conversion_seconds": round(time.time() - t0, 1),
        "peak_rss_gib": round(peak[0] / 2**30, 3),
        "all_tensors_passed_gate": not below,
        "below_gate_tensor_count": len(below),
        "below_gate_tensors": below[:20],
        "gate_policy": (
            "sem fallback raw: o último degrau é gravado mesmo reprovando o gate "
            "(RESCUE_LAST_RUNG) e o tensor sai com quality_flag. Rompe o "
            "invariante do v1 (§29.5) de propósito — verifique end-to-end antes "
            "de tratar o bundle como aprovado."
        ),
        "residencia": residency_report(total_out),
    }
    (out / "cascade_manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False))
    s = manifest["summary"]
    print(f"\nFonte {total_src / 1e9:.2f} GB -> bundle {total_out / 1e9:.2f} GB "
          f"({s['reducao_pct']}%, {s['bpw_medio']} bpw) em "
          f"{s['conversion_seconds']}s | RSS pico {s['peak_rss_gib']} GiB")
    if below:
        print(f"ATENCAO: {len(below)} tensor(es) ABAIXO do gate "
              f"(sem fallback raw): {', '.join(below[:5])}"
              f"{' ...' if len(below) > 5 else ''}")
        print("  Valide end-to-end antes de publicar este bundle.")
    else:
        print("Gate: todos os tensores 2D passaram")
    print(json.dumps(s["residencia"], indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
