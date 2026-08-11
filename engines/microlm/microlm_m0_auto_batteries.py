#!/usr/bin/env python3
"""MicroLM — bateria M0 da 7ª tecnologia (docs/C3_CONTRACTS_V1.md §22).

MicroLM v0.2 é um MODELO de referência (~22M params ativos fora as tabelas
engram; 27 camadas, mHC lanes/Sinkhorn, Engram hasheado, GQA janela+sinks com
RoPE cache-relativo, MLP Hadamard, init no-op exato) — NÃO é um otimizador.
`technology="MICROLM"` NUNCA é elegível na política do winner (contrato §1/§22).

Baterias (todas MEDIDAS; sem pytest — checagens embutidas):
    B0_MICROLM_NOOP_INIT        propriedade de init no-op exato na config de
                                referência: ‖logits − readout(emb)‖∞ ≤ 1e-4 e
                                contagem de params ativos em 20–24M
    P1_MICROLM_DECODE_PARITY    paridade decode_step vs forward de treino
                                (config SMALL; max abs diff ≤ 1e-3) + cache
                                limitado em geração longa (3× o limite)
    P1_MICROLM_DECODE_TOKS      tok/s REAL de decode em CPU na config de
                                referência; por run cronometrado: caches
                                NOVOS + prefill do prompt fora do cronômetro
                                (warmup) + 32 tokens novos medidos
                                (perf_counter_ns; mediana de 3 runs
                                idênticos) → candidate_tok_s medido;
                                baseline null (não há baseline comparável);
                                comparison_role=null
    P1_MICROLM_TRAINS_FROM_INIT 30 passos de Adam a partir do init exato
                                (config SMALL, SEM perturbação): loss final
                                < 0.8× inicial + wo/d3 saem do zero (prova de
                                que não há sela duplo-zero — FIX 2 do
                                CHANGES.md); curva de loss registrada
    P1_MICROLM_UNIT_CHECKS      espelho embutido das checagens-chave da suíte
                                (FWHT involução/norma, gate do engram vs zona
                                morta legada, matriz Sinkhorn duplamente
                                estocástica ~I, sinks visíveis pós-evicção):
                                contagem PASS/FAIL

Contrato de execução (espelha os irmãos engines/*):
  * importa model.py do MESMO diretório do script (o launcher Colab baixa
    model.py junto); probes: dir do script, /content, /content/microlm_run, cwd;
  * resultados: POST {"records": [...]} (schema v2, contrato §3/§7/§22) em
    RIFT_RESULTS_ENDPOINT com Authorization: Bearer $RIFT_INGEST_TOKEN;
  * artefatos em ./microlm_m0_test_output (no Colab: /content/...).

Honestidade de medição (docs/REAL_BENCHMARK_PROTOCOL_V3.md + contrato §3/§12):
  - latência/tok_s: time.perf_counter_ns com warmup; mediana de ≥3 medições;
  - *_ram_bytes de topo: SOMENTE pico VmRSS medido por fase (thread ~1ms);
  - comparison_role=null em TODOS os registros (modelo de referência sem
    baseline comparável); metrics.e2e.measured=true SÓ no DECODE_TOKS;
  - status FAIL nunca derruba o processo (exit 0 — as baterias reportam).

Segurança (contrato §5): endpoint HTTPS obrigatório e token de ingest com no
mínimo 32 caracteres (sem ambos a publicação é pulada com log claro); o token
é lido apenas de env vars / Colab Secrets, JAMAIS impresso ou gravado.

Sem pip install automático: dependências ausentes geram SystemExit com
instruções (torch é obrigatório).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

BENCHMARK_PROTOCOL = "MICROLM_M0_V1"
TECH_UPPER = "MICROLM"
MODEL_ID = "microlm/MicroLM-22M-v0.2"
SPEC_LABEL = "MicroLM v0.2 (referência)"
LATENCY_METHOD = "perf_counter_ns_median_v1"
RAM_METHOD = "proc_vmrss_sampling_per_phase_v1"
DECODE_RUN_BUDGET_S = 180.0  # guarda de tempo: run acima disso reduz p/ 16 tokens

# --- resolve model.py (dir do script primeiro; depois probes do Colab/cwd) ---
_HERE = Path(__file__).resolve().parent
for _cand in [_HERE, Path("/content"), Path("/content/microlm_run"), Path.cwd()]:
    try:
        if (_cand / "model.py").is_file():
            sys.path.insert(0, str(_cand))
            break
    except Exception:
        continue

# --- dependências (SEM pip automático: o launcher Colab instala antes) ---
try:
    import torch
    import torch.nn.functional as F
except ImportError:
    raise SystemExit(
        "[MICROLM] Dependência ausente: torch. Este script NÃO instala pacotes "
        "automaticamente — o launcher Colab deveria tê-lo instalado. Instale "
        "manualmente: pip install torch"
    )

try:
    from model import (
        AttentionCache,
        MHCLaneWrite,
        MicroLM,
        MicroLMConfig,
        fwht,
    )
except ImportError as _exc:
    raise SystemExit(
        f"[MICROLM] model.py não encontrado/importável ({_exc}). O launcher "
        "Colab baixa engines/microlm/model.py para o MESMO diretório deste "
        "script; localmente rode a partir de engines/microlm/."
    )

# Config SMALL de teste (espelho exato da suíte de referência test_model.py —
# barata para CPU; NÃO alterar sem sincronizar com a suíte)
SMALL = MicroLMConfig(
    vocab_size=512,
    d_model=128,
    n_layers=6,
    n_heads=4,
    n_kv_heads=2,
    head_dim=32,
    n_lanes=4,
    window=16,
    n_sink=4,
    engram_layers=(1, 3),
    engram_buckets=1 << 10,
    engram_hashes=2,
)


# ---------------------------------------------------------------------------
# Utilidades gerais (espelham c3_methodology/geyser_launcher)
# ---------------------------------------------------------------------------

def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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
    names = ("RIFT_INGEST_TOKEN", "RIFT_RESULTS_ENDPOINT")
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

    Retorna (resultado_fn, info) onde info = {max_bytes, mean_bytes, n_samples,
    method} ou None quando nenhuma medição real é possível (RAM de topo null).
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


def schema_v2_fields() -> Dict[str, Any]:
    """Campos obrigatórios do schema v2 (contrato §3) — device é sempre cpu."""
    torch_v = str(getattr(torch, "__version__", "unknown"))
    raw = f"{BENCHMARK_PROTOCOL}|{MODEL_ID}|cpu|{torch_v}"
    return {
        "schema_version": 2,
        "benchmark_protocol": BENCHMARK_PROTOCOL,
        "spec": SPEC_LABEL,
        "comparison_group_id": "cmp-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24],
        "comparison_context": {
            "protocol": BENCHMARK_PROTOCOL,
            "device": "cpu",
            "torch": torch_v,
            "python": platform.python_version(),
        },
        "implementation": {"kind": "REFERENCE_MEASURED", "native": False, "simulated": False},
    }


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
            "Authorization": f"Bearer {token}",  # nunca logado
            "User-Agent": "microlm-m0-battery/1.0",
        })
        with urlopen(req, timeout=60) as resp:
            print(f"[publish] HTTP {resp.status} battery={rec.get('battery_id')}")
    except Exception as exc:
        print(f"[publish] AVISO: {exc}")


class MicroLMRecorder:
    """Recorder incremental (espelha o C3Recorder): grava artefato por bateria,
    faz upsert no JSON agregado e publica cada registro assim que emitido.

    comparison_role=null e eligible_for_primary_ranking=false SEMPRE
    (contrato §22.1: MICROLM nunca entra na política do winner).
    """

    def __init__(self, out_dir: Path, *, run_id: str, schema_fields: Dict[str, Any],
                 publish_on: bool, endpoint: Optional[str] = None):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.batteries_dir = out_dir / "batteries"
        self.batteries_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = out_dir / "microlm_test_batteries.json"
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
        metrics: Optional[Dict[str, Any]] = None,
        notes: str = "",
        full_gate: Optional[bool] = None,
        ram_cand: Optional[Dict[str, Any]] = None,
        candidate_tok_s: Optional[float] = None,
        highlight: str = "",
    ) -> Dict[str, Any]:
        if full_gate is None:
            full_gate = status == "PASS"
        rec = {
            "timestamp_utc": utc(),
            "run_id": self.run_id,
            "technology": TECH_UPPER,
            "model_id": MODEL_ID,
            "battery_id": battery_id,
            "status": status,
            **self.schema_fields,
            # modelo de referência sem baseline comparável: nunca primary
            "comparison_role": None,
            "eligible_for_primary_ranking": False,
            "baseline_ram_bytes": None,
            "candidate_ram_bytes": ram_top_level(ram_cand),
            "baseline_disk_bytes": None,
            "candidate_disk_bytes": None,
            "baseline_tok_s": None,
            "candidate_tok_s": candidate_tok_s,
            "measurement_scope": scope,
            # quality.output null: as métricas específicas vivem em metrics
            "quality": {"full_local_gate_pass": bool(full_gate), "output": None},
            "metrics": metrics or {},
            "notes": notes[:1200],
        }
        path = self.batteries_dir / f"{self.run_id}__{battery_id}.json"
        path.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        # upsert pelo par (model_id, battery_id) no JSON agregado
        self.records = [
            r for r in self.records
            if r.get("battery_id") != battery_id or r.get("model_id") != MODEL_ID
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
# Helpers de modelo (portados da suíte test_model.py — sem pytest)
# ---------------------------------------------------------------------------

def _perturb(model: "MicroLM", std: float = 0.01, seed: int = 7) -> None:
    """Ruído determinístico nos parâmetros (porta o _perturb da suíte)."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(std * torch.randn(p.shape, generator=gen))


def _rand_ids(cfg: "MicroLMConfig", shape: Tuple[int, int], seed: int) -> "torch.Tensor":
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, cfg.vocab_size, shape, generator=gen)


# ---------------------------------------------------------------------------
# Baterias
# ---------------------------------------------------------------------------

def battery_noop_init(recorder: MicroLMRecorder) -> Optional["MicroLM"]:
    """B0: init no-op exato + contagem de params ativos na config de referência.

    Retorna o modelo de referência (reutilizado pelo DECODE_TOKS) ou None.
    """
    battery_id = "B0_MICROLM_NOOP_INIT"
    started = time.time()
    try:
        def _build_and_check():
            torch.manual_seed(0)
            cfg = MicroLMConfig()
            model = MicroLM(cfg)
            model.eval()
            ids = _rand_ids(cfg, (2, 16), seed=42)
            with torch.no_grad():
                logits = model(ids)
                h = model.embedding(ids)
                expected = (
                    model.final_norm(h) @ model.embedding.weight.T * model.logit_scale
                )
                linf = float((logits - expected).abs().max())
            active = int(model.active_parameter_count())
            total = int(sum(p.numel() for p in model.parameters()))
            return model, cfg, linf, active, total

        (model, cfg, linf, active, total), ram_info = measure_phase_ram(_build_and_check)
        linf_ok = linf <= 1e-4
        params_ok = 20_000_000 < active < 24_000_000
        status = "PASS" if (linf_ok and params_ok) else "FAIL"
        recorder.emit(
            battery_id, status,
            scope=(
                "MicroLM config de referência (27 camadas, d=512, V=8192, 4 lanes, "
                "janela 256 + 16 sinks) construída do zero em CPU; propriedade de "
                "init no-op exato MEDIDA: ‖logits − final_norm(embedding)·Eᵀ·"
                "logit_scale‖∞ sobre ids aleatórios de seed fixa; params ativos "
                "contados fora das tabelas engram; RAM = pico VmRSS da fase"
            ),
            metrics={
                "noop": {
                    "linf_logits_vs_readout_embedding": linf,
                    "threshold": 1e-4,
                    "pass": linf_ok,
                    "ids_shape": [2, 16],
                    "seed": 42,
                },
                "parameters": {
                    "active_parameter_count": active,
                    "active_range_expected": [20_000_000, 24_000_000],
                    "pass": params_ok,
                    "total_parameter_count_with_engram_tables": total,
                },
                "config": {
                    "n_layers": cfg.n_layers, "d_model": cfg.d_model,
                    "vocab_size": cfg.vocab_size, "n_lanes": cfg.n_lanes,
                    "window": cfg.window, "n_sink": cfg.n_sink,
                },
                "memory": {"candidate_phase": ram_info},
                "seconds": round(time.time() - started, 3),
            },
            ram_cand=ram_info,
            notes=(
                f"Init no-op exato: ‖Δ‖∞={linf:.3e} (gate ≤1e-4); params ativos="
                f"{active:,} (esperado 20–24M; tabelas engram fora da conta: "
                f"total={total:,}). Propriedade do CHANGES.md §3: no init o "
                "modelo inteiro é identidade — cada camada só escreve quando "
                "aprende algo."
            ),
            highlight=f"linf={linf:.1e} ativos={active/1e6:.1f}M",
        )
        return model  # modelo serve ao DECODE_TOKS mesmo com FAIL no gate
    except Exception as exc:
        traceback.print_exc()
        recorder.emit(
            battery_id, "FAIL",
            scope="B0 MicroLM (erro de execução)",
            metrics={"error": f"{type(exc).__name__}: {exc}"[:800],
                     "seconds": round(time.time() - started, 3)},
            notes=f"Falha ao construir/medir a config de referência: {exc}",
            highlight="erro",
        )
        return None


def battery_decode_parity(recorder: MicroLMRecorder) -> None:
    """P1: paridade decode_step vs forward de treino + cache limitado."""
    battery_id = "P1_MICROLM_DECODE_PARITY"
    started = time.time()
    try:
        def _run():
            torch.manual_seed(0)
            model = MicroLM(SMALL)
            _perturb(model, std=0.02, seed=7)
            model.eval()
            ids = _rand_ids(SMALL, (1, 10), seed=0)
            with torch.no_grad():
                ref = model(ids)
            caches = model.init_caches()
            diffs: List[float] = []
            for t in range(ids.shape[1]):
                step_logits = model.decode_step(ids[:, : t + 1], caches)
                diffs.append(float((step_logits[:, 0] - ref[:, t]).abs().max()))
            max_diff = max(diffs)

            # geração longa: cache limitado a n_sink + window após 3× o limite
            caches_long = model.init_caches()
            gen_ids = _rand_ids(SMALL, (1, 1), seed=3)
            limit = SMALL.n_sink + SMALL.window
            finite = True
            for _ in range(3 * limit):
                logits = model.decode_step(gen_ids, caches_long)
                if not bool(torch.isfinite(logits).all()):
                    finite = False
                    break
                nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
                gen_ids = torch.cat([gen_ids, nxt], dim=1)
            cache_lens = [len(c) for c in caches_long]
            bounded = all(n <= limit for n in cache_lens)
            return max_diff, diffs, limit, cache_lens, bounded, finite

        (max_diff, diffs, limit, cache_lens, bounded, finite), ram_info = \
            measure_phase_ram(_run)
        parity_ok = max_diff <= 1e-3
        status = "PASS" if (parity_ok and bounded and finite) else "FAIL"
        recorder.emit(
            battery_id, status,
            scope=(
                "Config SMALL (6 camadas, d=128, V=512) com ruído determinístico "
                "(_perturb std=0.02 seed=7); paridade MEDIDA: decode_step com "
                "AttentionCache vs forward de treino, 10 passos, max abs diff por "
                "passo; geração longa greedy de 3×(sinks+janela) passos com "
                "verificação de cache ≤ sinks+janela; RAM = pico VmRSS da fase"
            ),
            metrics={
                "parity": {
                    "max_abs_diff": max_diff,
                    "threshold": 1e-3,
                    "pass": parity_ok,
                    "per_step_abs_diff": diffs,
                    "steps": len(diffs),
                },
                "bounded_cache": {
                    "limit_n_sink_plus_window": limit,
                    "generation_steps": 3 * limit,
                    "cache_len_per_layer": cache_lens,
                    "max_cache_len": max(cache_lens) if cache_lens else None,
                    "pass": bounded,
                    "logits_finite": finite,
                },
                "memory": {"candidate_phase": ram_info},
                "seconds": round(time.time() - started, 3),
            },
            ram_cand=ram_info,
            # prosa condicionada às métricas: no caminho de falha o texto NÃO
            # pode afirmar o que as métricas estruturadas negam
            notes=(
                f"Paridade decode vs treino: max|Δ|={max_diff:.3e} (gate ≤1e-3) em "
                f"10 passos; geração longa de {3 * limit} passos "
                + (f"manteve cache ≤ {limit} em todas as camadas" if bounded
                   else f"EXCEDEU o limite de cache {limit} em ao menos uma camada")
                + f" (max={max(cache_lens)}) com logits "
                + ("finitos" if finite else "NÃO finitos")
                + ". RoPE cache-relativo pós-evicção (CHANGES.md §4)."
            ),
            highlight=f"maxΔ={max_diff:.1e} cache≤{max(cache_lens)}/{limit}",
        )
    except Exception as exc:
        traceback.print_exc()
        recorder.emit(
            battery_id, "FAIL",
            scope="P1 MicroLM paridade de decode (erro de execução)",
            metrics={"error": f"{type(exc).__name__}: {exc}"[:800],
                     "seconds": round(time.time() - started, 3)},
            notes=f"Falha na bateria de paridade: {exc}",
            highlight="erro",
        )


def battery_decode_toks(recorder: MicroLMRecorder, model: Optional["MicroLM"]) -> None:
    """P1: tok/s REAL de decode greedy em CPU na config de referência (22M)."""
    battery_id = "P1_MICROLM_DECODE_TOKS"
    started = time.time()
    try:
        if model is None:
            recorder.emit(
                battery_id, "FAIL",
                scope="P1 MicroLM tok/s de decode (indisponível)",
                metrics={"error": "modelo de referência não construído no B0",
                         "seconds": round(time.time() - started, 3)},
                notes="B0 falhou antes de construir o modelo de referência — sem medição.",
                highlight="sem modelo",
            )
            return
        cfg = model.cfg
        model.eval()

        def _measure():
            with torch.no_grad():
                ids = _rand_ids(cfg, (1, 4), seed=11)
                prompt_tokens = ids.shape[1]
                runs: List[Dict[str, Any]] = []
                tokens_per_run = 32
                reduced = False
                for _ in range(3):
                    # caches NOVOS por run: 3 medições idênticas e
                    # independentes sob o MESMO protocolo (§3/§22.2) — sem
                    # reset, cada run mediria uma carga de atenção diferente.
                    caches = model.init_caches()
                    # prefill do prompt token a token FORA da região
                    # cronometrada (o prefill é o warmup ≥2 passos; não
                    # conta no tok/s)
                    for t in range(prompt_tokens):
                        last_logits = model.decode_step(ids[:, : t + 1], caches)
                    # materializa o 1º token novo ANTES do cronômetro: cada
                    # decode_step processa UM token novo (embute cur[:, -1:]
                    # e faz um único cache.append) — sem isso o último token
                    # do prompt seria reprocessado e duplicado no cache.
                    nxt = last_logits[:, -1].argmax(dim=-1, keepdim=True)
                    cur = torch.cat([ids, nxt], dim=1)
                    t0 = time.perf_counter_ns()
                    for _ in range(tokens_per_run):
                        last_logits = model.decode_step(cur, caches)
                        nxt = last_logits[:, -1].argmax(dim=-1, keepdim=True)
                        cur = torch.cat([cur, nxt], dim=1)
                    seconds = (time.perf_counter_ns() - t0) / 1e9
                    finite = bool(torch.isfinite(last_logits).all())
                    runs.append({
                        "tokens": tokens_per_run,
                        "seconds": round(seconds, 4),
                        "tok_s": tokens_per_run / seconds if seconds > 0 else 0.0,
                        "logits_finite": finite,
                    })
                    if seconds > DECODE_RUN_BUDGET_S and tokens_per_run != 16:
                        tokens_per_run = 16
                        reduced = True
                return prompt_tokens, runs, reduced

        (prompt_tokens, runs, reduced), ram_info = measure_phase_ram(_measure)
        tok_s_values = [r["tok_s"] for r in runs]
        candidate_tok_s = float(statistics.median(tok_s_values))
        all_finite = all(r["logits_finite"] for r in runs)
        status = "PASS" if (candidate_tok_s > 0 and all_finite) else "FAIL"
        budget_note = (
            f" Guarda de tempo: um run de 32 tokens excedeu {DECODE_RUN_BUDGET_S:.0f}s — "
            "runs seguintes reduzidos para 16 tokens (tok/s normalizado por run); "
            "DESVIO do protocolo §22.2 (runs de 32 tokens) registrado em "
            "metrics.decode.protocol_deviation."
            if reduced else ""
        )
        recorder.emit(
            battery_id, status,
            scope=(
                "tok/s REAL de decode greedy token a token em CPU (Python de "
                "referência, config 22M): por run cronometrado, caches NOVOS + "
                "prefill do prompt de 4 tokens fora da região cronometrada "
                "(warmup) + 32 tokens novos cronometrados (perf_counter_ns; "
                "mediana do tok/s de 3 runs idênticos e independentes) → "
                "candidate_tok_s; baseline_tok_s null — modelo de referência "
                "sem baseline comparável; RAM = pico VmRSS da fase de decode"
            ),
            metrics={
                "e2e": {"measured": True, "scope": "python_reference_cpu_decode"},
                "decode": {
                    "candidate_tok_s_median": candidate_tok_s,
                    "runs": runs,
                    "prompt_tokens_per_run": prompt_tokens,
                    "cache_reset_per_run": True,
                    "budget_seconds_per_run": DECODE_RUN_BUDGET_S,
                    "budget_reduced_to_16_tokens": reduced,
                    "protocol_deviation": reduced,
                    "method": LATENCY_METHOD,
                    "device": "cpu",
                    "greedy": True,
                },
                "memory": {"candidate_phase": ram_info},
                "seconds": round(time.time() - started, 3),
            },
            ram_cand=ram_info,
            candidate_tok_s=candidate_tok_s,
            notes=(
                f"Decode CPU da referência 22M: {candidate_tok_s:.3f} tok/s "
                f"(mediana de {len(runs)} runs; wall-clock Python — NÃO representa "
                "kernel nativo). Sem baseline comparável (é o próprio modelo de "
                f"referência), então baseline_tok_s=null.{budget_note}"
            ),
            highlight=f"{candidate_tok_s:.2f} tok/s (CPU)",
        )
    except Exception as exc:
        traceback.print_exc()
        recorder.emit(
            battery_id, "FAIL",
            scope="P1 MicroLM tok/s de decode (erro de execução)",
            metrics={"error": f"{type(exc).__name__}: {exc}"[:800],
                     "seconds": round(time.time() - started, 3)},
            notes=f"Falha na medição de tok/s: {exc}",
            highlight="erro",
        )


def battery_trains_from_init(recorder: MicroLMRecorder) -> None:
    """P1: 30 passos de Adam a partir do init EXATO (sem perturbação)."""
    battery_id = "P1_MICROLM_TRAINS_FROM_INIT"
    started = time.time()
    try:
        def _train():
            torch.manual_seed(0)
            model = MicroLM(SMALL)
            ids = _rand_ids(SMALL, (2, 12), seed=0)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            losses: List[float] = []
            for _ in range(30):
                logits = model(ids)
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, SMALL.vocab_size),
                    ids[:, 1:].reshape(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss))
            wo_max = float(model.layers[0].attn.wo.weight.abs().max())
            d3_max = float(model.layers[0].mlp.d3.abs().max())
            return losses, wo_max, d3_max

        (losses, wo_max, d3_max), ram_info = measure_phase_ram(_train)
        loss_ok = losses[-1] < 0.8 * losses[0]
        weights_ok = wo_max > 0.0 and d3_max > 0.0
        status = "PASS" if (loss_ok and weights_ok) else "FAIL"
        recorder.emit(
            battery_id, status,
            scope=(
                "Config SMALL a partir do init no-op EXATO (sem perturbação): 30 "
                "passos de Adam lr=1e-3 com cross-entropy next-token em ids fixos; "
                "gate: loss final < 0.8× inicial E wo/d3 da camada 0 saem do zero "
                "(prova de que não há sela duplo-zero — FIX 2 do CHANGES.md); "
                "RAM = pico VmRSS da fase"
            ),
            metrics={
                "training": {
                    "optimizer": "Adam",
                    "lr": 1e-3,
                    "steps": 30,
                    "loss_initial": losses[0],
                    "loss_final": losses[-1],
                    "loss_ratio": losses[-1] / losses[0] if losses[0] else None,
                    "threshold_ratio": 0.8,
                    "loss_curve": [round(v, 6) for v in losses],
                    "pass": loss_ok,
                },
                "escaped_saddle": {
                    "wo_abs_max_after": wo_max,
                    "d3_abs_max_after": d3_max,
                    "pass": weights_ok,
                    "nota": "g=1 no mHC write mantém o no-op do init e abre o "
                            "caminho de gradiente (CHANGES.md FIX 2)",
                },
                "memory": {"candidate_phase": ram_info},
                "seconds": round(time.time() - started, 3),
            },
            ram_cand=ram_info,
            notes=(
                f"Treina do init exato: loss {losses[0]:.4f} → {losses[-1]:.4f} "
                f"(razão {losses[-1] / losses[0]:.3f}; gate <0.8) em 30 passos de "
                f"Adam; wo|max|={wo_max:.3e} e d3|max|={d3_max:.3e} saíram do zero "
                "— sem sela duplo-zero."
            ),
            highlight=f"loss {losses[0]:.2f}→{losses[-1]:.2f}",
        )
    except Exception as exc:
        traceback.print_exc()
        recorder.emit(
            battery_id, "FAIL",
            scope="P1 MicroLM treino do init (erro de execução)",
            metrics={"error": f"{type(exc).__name__}: {exc}"[:800],
                     "seconds": round(time.time() - started, 3)},
            notes=f"Falha na bateria de treino: {exc}",
            highlight="erro",
        )


def battery_unit_checks(recorder: MicroLMRecorder) -> None:
    """P1: espelho embutido das checagens-chave da suíte (contagem PASS/FAIL)."""
    battery_id = "P1_MICROLM_UNIT_CHECKS"
    started = time.time()
    try:
        def _checks() -> List[Dict[str, Any]]:
            details: List[Dict[str, Any]] = []

            # 1) FWHT: involução (fwht(fwht(x)) == x)
            torch.manual_seed(0)
            x = torch.randn(3, 5, 128)
            invol_diff = float((fwht(fwht(x)) - x).abs().max())
            details.append({
                "check": "fwht_involution",
                "measured": {"max_abs_diff": invol_diff},
                "threshold": 1e-5,
                "pass": invol_diff <= 1e-5,
            })

            # 2) FWHT: preserva a norma (transformada ortonormal)
            torch.manual_seed(1)
            xn = torch.randn(4, 128)
            norm_diff = float((xn.norm(dim=-1) - fwht(xn).norm(dim=-1)).abs().max())
            details.append({
                "check": "fwht_preserves_norm",
                "measured": {"max_abs_norm_diff": norm_diff},
                "threshold": 1e-5,
                "pass": norm_diff <= 1e-5,
            })

            # 3) Gate do engram: span amplo vs zona morta legada (/√d)
            import math as _math
            cos = torch.tensor([-1.0, 1.0])
            legacy = torch.sigmoid(cos / _math.sqrt(512))
            legacy_span = float(legacy[1] - legacy[0])
            fixed = torch.sigmoid(SMALL.gate_tau_init * cos + SMALL.gate_beta_init)
            fixed_span = float(fixed[1] - fixed[0])
            details.append({
                "check": "engram_gate_span_vs_legacy_dead_zone",
                "measured": {"legacy_span": legacy_span, "fixed_span": fixed_span},
                "threshold": {"legacy_span_lt": 0.05, "fixed_span_gt": 0.5},
                "pass": legacy_span < 0.05 and fixed_span > 0.5,
            })

            # 4) Sinkhorn: matriz de mistura duplamente estocástica ~identidade
            write = MHCLaneWrite(SMALL)
            p = write.mixing_matrix()
            row_dev = float((p.sum(dim=1) - 1.0).abs().max())
            col_dev = float((p.sum(dim=0) - 1.0).abs().max())
            diag_min = float(p.diagonal().min())
            details.append({
                "check": "sinkhorn_doubly_stochastic_near_identity",
                "measured": {"row_sum_dev_max": row_dev,
                             "col_sum_dev_max": col_dev,
                             "diag_min": diag_min},
                "threshold": {"row_col_dev_le": 1e-4, "diag_min_gt": 0.99},
                "pass": row_dev <= 1e-4 and col_dev <= 1e-4 and diag_min > 0.99,
            })

            # 5) Sinks visíveis após evicção (cache guarda os âncora intactos)
            torch.manual_seed(0)
            cache = AttentionCache(SMALL)
            b, kvh, hd = 1, SMALL.n_kv_heads, SMALL.head_dim
            marker = torch.full((b, SMALL.n_sink, kvh, hd), 9.0)
            cache.append(marker, marker)
            for _ in range(SMALL.window * 2):
                step = torch.randn(b, 1, kvh, hd)
                cache.append(step, step)
            expected_len = SMALL.n_sink + SMALL.window
            sink_ok = bool(torch.equal(cache.k[:, : SMALL.n_sink], marker))
            details.append({
                "check": "attention_sink_tokens_visible_beyond_window",
                "measured": {"cache_len": len(cache),
                             "expected_len": expected_len,
                             "sink_tokens_preserved": sink_ok},
                "threshold": {"cache_len_eq": expected_len, "sink_preserved": True},
                "pass": len(cache) == expected_len and sink_ok,
            })
            return details

        details, ram_info = measure_phase_ram(_checks)
        passed = sum(1 for d in details if d["pass"])
        failed = len(details) - passed
        status = "PASS" if failed == 0 else "FAIL"
        failing = [d["check"] for d in details if not d["pass"]]
        recorder.emit(
            battery_id, status,
            scope=(
                "Espelho embutido (sem pytest) das checagens-chave da suíte de "
                "referência: FWHT involução e preservação de norma, gate do engram "
                "com span amplo vs zona morta legada /√d, matriz de mistura "
                "Sinkhorn duplamente estocástica ~identidade, sinks visíveis após "
                "evicção da janela; cada checagem MEDIDA com números registrados"
            ),
            metrics={
                "unit_checks": {
                    "passed": passed,
                    "failed": failed,
                    "details": details,
                },
                "memory": {"candidate_phase": ram_info},
                "seconds": round(time.time() - started, 3),
            },
            ram_cand=ram_info,
            notes=(
                f"Checagens de unidade embutidas: {passed}/{len(details)} PASS"
                + (f"; falhas: {', '.join(failing)}" if failing else "")
                + ". Cobrem os FIXes 1–4 do CHANGES.md (gate do engram, Sinkhorn "
                  "quase-identidade, FWHT ortonormal, sinks pós-evicção)."
            ),
            highlight=f"{passed}/{len(details)} checagens",
        )
    except Exception as exc:
        traceback.print_exc()
        recorder.emit(
            battery_id, "FAIL",
            scope="P1 MicroLM checagens de unidade (erro de execução)",
            metrics={"error": f"{type(exc).__name__}: {exc}"[:800],
                     "seconds": round(time.time() - started, 3)},
            notes=f"Falha nas checagens de unidade: {exc}",
            highlight="erro",
        )


# ---------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MicroLM — bateria M0 da 7ª tecnologia (contrato §22)")
    p.add_argument("--out", default="microlm_m0_test_output")
    p.add_argument("--publish", default="on", choices=["on", "off"])
    p.add_argument("--results-endpoint", default=None,
                   help="URL HTTPS /api/results (default: env)")
    values = sys.argv[1:] if argv is None else list(argv)
    return p.parse_args(without_ipykernel_connection_args(values))


def main(argv=None) -> int:
    args = parse_args(argv)
    bootstrap_colab_secrets()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id()
    recorder = MicroLMRecorder(
        out_dir, run_id=run_id, schema_fields=schema_v2_fields(),
        publish_on=args.publish != "off", endpoint=args.results_endpoint,
    )

    print("=" * 96)
    print(f"MICROLM — bateria M0 ({BENCHMARK_PROTOCOL}) | model={MODEL_ID} | "
          f"device=cpu | run={run_id}")
    print("=" * 96)

    ref_model = battery_noop_init(recorder)
    battery_decode_parity(recorder)
    battery_decode_toks(recorder, ref_model)
    del ref_model
    battery_trains_from_init(recorder)
    battery_unit_checks(recorder)

    # gain report (resumo p/ dashboard e auditoria)
    by_id = {r.get("battery_id"): r for r in recorder.records
             if r.get("run_id") == run_id}

    def _m(bid: str, *path):
        cur: Any = (by_id.get(bid) or {}).get("metrics") or {}
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                return None
            cur = cur[key]
        return cur

    summary = {
        "run_id": run_id,
        "technology": TECH_UPPER,
        "model_id": MODEL_ID,
        "benchmark_protocol": BENCHMARK_PROTOCOL,
        "device": "cpu",
        "noop_linf": _m("B0_MICROLM_NOOP_INIT", "noop",
                        "linf_logits_vs_readout_embedding"),
        "active_parameter_count": _m("B0_MICROLM_NOOP_INIT", "parameters",
                                     "active_parameter_count"),
        "decode_parity_max_abs_diff": _m("P1_MICROLM_DECODE_PARITY", "parity",
                                         "max_abs_diff"),
        "decode_cpu_tok_s": _m("P1_MICROLM_DECODE_TOKS", "decode",
                               "candidate_tok_s_median"),
        "train_loss_initial": _m("P1_MICROLM_TRAINS_FROM_INIT", "training",
                                 "loss_initial"),
        "train_loss_final": _m("P1_MICROLM_TRAINS_FROM_INIT", "training",
                               "loss_final"),
        "unit_checks_passed": _m("P1_MICROLM_UNIT_CHECKS", "unit_checks", "passed"),
        "unit_checks_failed": _m("P1_MICROLM_UNIT_CHECKS", "unit_checks", "failed"),
        "records_total": len(recorder.summary_rows),
        "records_pass": sum(1 for r in recorder.summary_rows if r["status"] == "PASS"),
        "generated_at": utc(),
    }
    gain_path = out_dir / "microlm_gain_report.json"
    gain_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")

    # tabela final PT-BR (espelha os irmãos)
    print()
    print("=" * 96)
    print(f"MICROLM — bateria M0 | model={MODEL_ID} | device=cpu")
    print("=" * 96)
    print(f"{'battery_id':<34} {'status':<10} destaque")
    print("-" * 96)
    for row in recorder.summary_rows:
        print(f"{row['battery_id']:<34} {row['status']:<10} {row['highlight']}")
    print("-" * 96)
    tok_s = summary.get("decode_cpu_tok_s")
    if tok_s is not None:
        print(f"Tok/s decode CPU (REAL) : {tok_s:.3f} (Python de referência; "
              "sem baseline comparável)")
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
        _rc = 0  # baterias reportam FAIL nos registros; infraestrutura não derruba
    sys.exit(_rc)
