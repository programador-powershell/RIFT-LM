#!/usr/bin/env python3
"""compare_generations_publisher.py — conversor + publisher da comparação de gerações.

Contrato: docs/C3_CONTRACTS_V1.md §18.2. Converte o artefato
`compare_generations_report.json` (formato `{model, target_linear,
schemes_tensor{TECH...}, e2e{ORIGINAL+TECH...}, ceilings}`) em registros
schema v2 (`CMP_<TECH>_GENERATIONS`, um por tecnologia do bloco `e2e`,
pulando ORIGINAL) e publica em `POST {records:[...]}` no ingest.

Hardening padrão dos publishers (scripts/security_check.mjs, heurística c):
o POST exige endpoint HTTPS e token RIFT_INGEST_TOKEN com tamanho >= 32
caracteres; qualquer ausência é pulada COM LOG (nunca quebra o run).
Fallback de segredo: Colab Secrets (google.colab.userdata) guardado por
try-import. Stdlib-only — sem torch/numpy/requests.

Honestidade (§18.2): PPL/top1/KL/tempo são MEDIDOS na comparação e2e real;
os tetos de tok/s (`ceilings`) são PROJETADOS (lei de banda) e ficam apenas
em `metrics.compare` com rótulo — tok/s, RAM e disco de nível superior
permanecem null; `comparison_role=null` e
`eligible_for_primary_ranking=false` (não entra na política do winner).

Uso:
    python compare_generations_publisher.py [report.json]
        [--results-endpoint URL] [--publish on|off] [--run-id ID]
    python compare_generations_publisher.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

BENCHMARK_PROTOCOL = "COMPARE_GENERATIONS_V1"
SPEC = "compare_generations v1"
DEFAULT_REPORT_PATH = "./compare_generations_report.json"
DEFAULT_ENDPOINT = "https://rift-lm.vercel.app/api/results"
RECORDS_BASENAME = "compare_generations_records.json"
GEN_SAMPLE_MAX_CHARS = 200
CEILINGS_LABEL = (
    "PROJETADO — teto teórico de tok/s pela lei de banda; NÃO é medição"
)
# Enum de tecnologias aceito pelo ingest (api/results.mjs). Tecnologias fora
# do enum são convertidas e gravadas localmente, mas EXCLUÍDAS do POST (um
# registro rejeitado derrubaria o lote inteiro no ingest).
KNOWN_TECHS = ("RIFT", "CASCADE", "AETHER", "SPECTRA", "WINNER", "GEYSER")


def log(message: str) -> None:
    print(f"[cmp-publisher] {message}")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _finite(value):
    """Número finito ou None (bool não é número aqui; NaN/inf viram None)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _sanitize_tech(raw) -> str:
    """Nome de tecnologia -> fragmento de battery_id ([A-Z0-9_])."""
    text = str(raw or "").strip().upper()
    return "".join(ch if (ch.isascii() and ch.isalnum()) else "_" for ch in text)


def best_output_cos(node):
    """Maior `output_cos` finito derivável do bloco schemes_tensor da tech.

    Percorre dicts aninhados (cobre a ladder da AETHER); None se nada houver.
    """
    best = None
    stack = [node]
    while stack:
        current = stack.pop()
        if not isinstance(current, dict):
            continue
        for key, value in current.items():
            if isinstance(value, dict):
                stack.append(value)
            elif key == "output_cos":
                number = _finite(value)
                if number is not None and (best is None or number > best):
                    best = number
    return best


def comparison_group_id(model_id: str) -> str:
    """cmp-<sha256[:24]> de 'COMPARE_GENERATIONS_V1|<model>|cpu|-' (§18.2)."""
    raw = f"{BENCHMARK_PROTOCOL}|{model_id}|cpu|-"
    return "cmp-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def build_comparison_context(report: dict, model_id: str) -> dict:
    """Contexto a partir dos campos do relatório quando presentes.

    Os campos publisher_* descrevem a máquina que CONVERTEU/publicou — o
    ambiente de medição não consta no artefato, e não é inventado aqui.
    """
    context = {
        "protocol": BENCHMARK_PROTOCOL,
        "device": "cpu",
        "context_resolution": "REPORT_LEVEL",
        "source": "compare_generations_publisher.py",
        "publisher_python": platform.python_version(),
        "publisher_platform": platform.platform(),
        "publisher_machine": platform.machine(),
        "nota": (
            "ambiente de MEDIÇÃO não consta no relatório; campos "
            "publisher_* descrevem a máquina que converteu/publicou"
        ),
    }
    if model_id:
        context["model_id"] = model_id
    for key in ("target_linear", "shape", "labels"):
        if key in report:
            context[key] = report[key]
    return context


def convert_report(report: dict, run_id: str, timestamp_utc: str) -> list:
    """Converte o relatório em registros schema v2 (um por tech do e2e)."""
    if not isinstance(report, dict):
        raise ValueError("relatório precisa ser um objeto JSON")
    e2e = report.get("e2e")
    if not isinstance(e2e, dict) or not e2e:
        raise ValueError("relatório sem bloco e2e — nada a converter")

    model_id = str(report.get("model") or "").strip()
    schemes_tensor = report.get("schemes_tensor")
    if not isinstance(schemes_tensor, dict):
        schemes_tensor = {}
        log("AVISO: relatório sem schemes_tensor — quality.output.cosine=null")
    ceilings = report.get("ceilings")
    group_id = comparison_group_id(model_id)
    context = build_comparison_context(report, model_id)

    original = None
    for key, value in e2e.items():
        if _sanitize_tech(key) == "ORIGINAL" and isinstance(value, dict):
            original = value
            break
    ppl_original = _finite(original.get("ppl_calib")) if original else None
    if ppl_original is None:
        log("AVISO: e2e.ORIGINAL.ppl_calib ausente — gate de PPL não "
            "verificável; status será EXPERIMENTAL_FAIL")

    records = []
    for raw_tech, entry in e2e.items():
        tech = _sanitize_tech(raw_tech)
        if tech == "ORIGINAL":
            continue
        if not isinstance(entry, dict):
            log(f"AVISO: e2e.{raw_tech} não é objeto — ignorado")
            continue
        if not tech:
            log(f"AVISO: tecnologia sem nome utilizável em e2e ({raw_tech!r}) — ignorada")
            continue

        top1 = _finite(entry.get("top1_agreement_calib"))
        ppl = _finite(entry.get("ppl_calib"))
        mean_kl = _finite(entry.get("mean_kl_calib"))
        seconds = _finite(entry.get("seconds"))
        bits_eff = _finite(entry.get("bits_eff_blocos"))
        linears = _finite(entry.get("linears_patched"))
        linears = int(linears) if linears is not None else None

        # Gate §18: PASS sse top1>=0.70 E ppl<=1.5*ppl_original (ambos
        # verificáveis); qualquer métrica ausente reprova por honestidade.
        gate_pass = (
            top1 is not None and ppl is not None
            and ppl_original is not None and ppl_original > 0
            and top1 >= 0.70 and ppl <= 1.5 * ppl_original
        )
        status = "PASS" if gate_pass else "EXPERIMENTAL_FAIL"

        gen_sample = entry.get("gen_sample")
        if isinstance(gen_sample, str) and len(gen_sample) > GEN_SAMPLE_MAX_CHARS:
            gen_sample = gen_sample[:GEN_SAMPLE_MAX_CHARS]
        elif not isinstance(gen_sample, str):
            gen_sample = None

        ppl_ratio = None
        if ppl is not None and ppl_original is not None and ppl_original > 0:
            ppl_ratio = round(ppl / ppl_original, 6)

        scheme_tensor = schemes_tensor.get(raw_tech)
        if scheme_tensor is None:
            scheme_tensor = schemes_tensor.get(tech)

        notes = None
        if isinstance(scheme_tensor, dict) and isinstance(scheme_tensor.get("nota"), str):
            notes = scheme_tensor["nota"]
        elif isinstance(entry.get("nota"), str):
            notes = entry["nota"]

        n_linears = linears if linears is not None else "N?"
        measurement_scope = (
            f"comparação e2e real com {n_linears} Linear patchadas; "
            "PPL/top1/KL medidos; tetos são PROJETADOS"
        )

        compare = {
            "top1_agreement": top1,
            "mean_kl": mean_kl,
            "ppl": ppl,
            "ppl_original": ppl_original,
            "ppl_ratio": ppl_ratio,
            "gen_sample": gen_sample,
            "bits_eff_blocos": bits_eff,
            "seconds": seconds,
            "linears_patched": linears,
            "scheme_tensor": scheme_tensor,
            "ceilings_label": CEILINGS_LABEL,
            "ceilings": ceilings,
        }

        record = {
            "schema_version": 2,
            "timestamp_utc": timestamp_utc,
            "run_id": run_id,
            "spec": SPEC,
            "technology": tech,
            "battery_id": f"CMP_{tech}_GENERATIONS",
            "benchmark_protocol": BENCHMARK_PROTOCOL,
            "comparison_role": None,
            "comparison_group_id": group_id,
            "comparison_context": context,
            "implementation": {
                "kind": "REFERENCE_MEASURED",
                "native": False,
                "simulated": False,
            },
            "eligible_for_primary_ranking": False,
            "status": status,
            "baseline_tok_s": None,
            "candidate_tok_s": None,
            "baseline_ram_bytes": None,
            "candidate_ram_bytes": None,
            "baseline_disk_bytes": None,
            "candidate_disk_bytes": None,
            "quality": {
                "full_local_gate_pass": status == "PASS",
                "output": {
                    "cosine": best_output_cos(scheme_tensor),
                    "nrmse": None,
                },
            },
            "metrics": {"compare": compare},
            "gains": {},
            "measurement_scope": measurement_scope,
            "notes": notes,
        }
        if model_id:
            record["model_id"] = model_id
        records.append(record)

    if not records:
        raise ValueError("nenhuma tecnologia utilizável no bloco e2e (além de ORIGINAL)")
    return records


# ------------------------------ publicação -----------------------------------

def resolve_ingest_token() -> str:
    """RIFT_INGEST_TOKEN por env; fallback Colab Secrets (try-import guardado).

    O enforcement (endpoint HTTPS + token >= 32 chars) acontece em
    publish_records(); token nunca é logado.
    """
    token = (os.environ.get("RIFT_INGEST_TOKEN") or "").strip()
    if token:
        return token
    try:
        from google.colab import userdata  # type: ignore

        return str(userdata.get("RIFT_INGEST_TOKEN") or "").strip()
    except Exception:
        return ""


def publish_records(records: list, endpoint: str) -> bool:
    """POST {records:[...]} em UMA requisição; nunca lança (avisos com log)."""
    token = resolve_ingest_token()
    if len(token) < 32:
        log("publicação PULADA: RIFT_INGEST_TOKEN ausente ou curto (<32 chars) "
            "— registros locais preservados")
        return False
    if not str(endpoint).lower().startswith("https://"):
        log(f"publicação PULADA: endpoint não-HTTPS bloqueado: {endpoint}")
        return False

    to_send = [r for r in records if r.get("technology") in KNOWN_TECHS]
    excluded = [r["battery_id"] for r in records
                if r.get("technology") not in KNOWN_TECHS]
    if excluded:
        log("AVISO: fora do enum do ingest, excluídos do POST: "
            + ", ".join(excluded))
    if not to_send:
        log("publicação PULADA: nenhum registro com tecnologia aceita pelo ingest")
        return False

    body = json.dumps({"records": to_send}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",  # nunca logado
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            log(f"ingest respondeu HTTP {response.status} "
                f"({len(to_send)} registro(s) enviados)")
            return True
    except urllib.error.HTTPError as exc:
        log(f"AVISO: HTTP {exc.code} ao publicar — registros locais preservados")
    except Exception as exc:  # noqa: BLE001 — publicação nunca derruba o run
        log(f"AVISO: falha de rede ao publicar ({type(exc).__name__}) "
            "— registros locais preservados")
    return False


# ------------------------------- selftest ------------------------------------

_SELFTEST_REPORT = {
    "model": "org-teste/modelo-mini",
    "target_linear": "model.layers.0.mlp.gate_proj",
    "shape": [8, 4],
    "labels": {"qualidade/agreement/ppl": "MEDIDO", "tetos_tok_s": "PROJETADO"},
    "schemes_tensor": {
        "CASCADE": {
            "F0_int4g32": {"output_cos": 0.9985, "output_nrmse": 0.054},
            "F0_r16_sempre": {"output_cos": 0.9986, "output_nrmse": 0.0528},
            "nota": "nota CASCADE de teste",
        },
        "AETHER": {
            "base_sign_ternary": {"output_cos": 0.7862},
            "tadds_ladder_rank": {"16": {"output_cos": 0.909}},
            "nota": "nota AETHER de teste",
        },
    },
    "e2e": {
        "ORIGINAL": {"ppl_calib": 50.0, "gen_sample": "texto original"},
        "CASCADE": {
            "linears_patched": 168,
            "bits_eff_blocos": 5.393,
            "top1_agreement_calib": 0.7422,
            "mean_kl_calib": 0.148,
            "ppl_calib": 56.0,
            "gen_sample": "x" * 300,
            "seconds": 17.2,
        },
        "AETHER": {
            "linears_patched": 168,
            "bits_eff_blocos": 2.393,
            "top1_agreement_calib": 0.0,
            "mean_kl_calib": 14.1,
            "ppl_calib": 41542308.0,
            "gen_sample": "lixo",
            "seconds": 15.0,
        },
    },
    "ceilings": {
        "label": "PROJETADO — teto de exemplo",
        "bw_gbps_medida": 11.0,
        "por_esquema_tok_s": {"CASCADE_int4": 29.25},
    },
}


def selftest() -> int:
    """Valida a conversão com fixture embutida — SEM rede e SEM filesystem."""
    run_id = "cmp-selftest"
    timestamp = "2026-08-10T00:00:00Z"
    records = convert_report(_SELFTEST_REPORT, run_id, timestamp)
    try:
        assert len(records) == 2, f"esperava 2 registros, veio {len(records)}"
        by_id = {r["battery_id"]: r for r in records}
        cascade = by_id["CMP_CASCADE_GENERATIONS"]
        aether = by_id["CMP_AETHER_GENERATIONS"]

        # Gate §18: CASCADE passa (0.7422>=0.70 e 56<=1.5*50); AETHER reprova.
        assert cascade["status"] == "PASS", cascade["status"]
        assert aether["status"] == "EXPERIMENTAL_FAIL", aether["status"]
        assert cascade["quality"]["full_local_gate_pass"] is True
        assert aether["quality"]["full_local_gate_pass"] is False

        for record in records:
            assert record["schema_version"] == 2
            assert record["run_id"] == run_id
            assert record["timestamp_utc"] == timestamp
            assert record["spec"] == SPEC
            assert record["benchmark_protocol"] == BENCHMARK_PROTOCOL
            assert record["comparison_role"] is None
            assert record["eligible_for_primary_ranking"] is False
            assert record["gains"] == {}
            assert record["model_id"] == "org-teste/modelo-mini"
            for field in ("baseline_tok_s", "candidate_tok_s",
                          "baseline_ram_bytes", "candidate_ram_bytes",
                          "baseline_disk_bytes", "candidate_disk_bytes"):
                assert record[field] is None, f"{field} deveria ser null"
            expected_group = "cmp-" + hashlib.sha256(
                "COMPARE_GENERATIONS_V1|org-teste/modelo-mini|cpu|-".encode("utf-8")
            ).hexdigest()[:24]
            assert record["comparison_group_id"] == expected_group
            compare = record["metrics"]["compare"]
            assert compare["ceilings"] == _SELFTEST_REPORT["ceilings"]
            assert compare["ceilings_label"].startswith("PROJETADO")
            assert "PROJETADOS" in record["measurement_scope"]
            assert "168" in record["measurement_scope"]

        cmp_cascade = cascade["metrics"]["compare"]
        assert cmp_cascade["top1_agreement"] == 0.7422
        assert cmp_cascade["ppl"] == 56.0
        assert cmp_cascade["ppl_original"] == 50.0
        assert abs(cmp_cascade["ppl_ratio"] - 1.12) < 1e-9
        assert len(cmp_cascade["gen_sample"]) == GEN_SAMPLE_MAX_CHARS
        assert cmp_cascade["scheme_tensor"] == _SELFTEST_REPORT["schemes_tensor"]["CASCADE"]
        assert cascade["quality"]["output"]["cosine"] == 0.9986
        assert cascade["quality"]["output"]["nrmse"] is None
        assert cascade["notes"] == "nota CASCADE de teste"

        # AETHER: melhor cosine vem da ladder aninhada.
        assert aether["quality"]["output"]["cosine"] == 0.909
        assert aether["notes"] == "nota AETHER de teste"
        assert aether["metrics"]["compare"]["linears_patched"] == 168
    except AssertionError as exc:
        log(f"SELFTEST FALHOU: {exc}")
        return 1
    log("SELFTEST OK: 2 registros (CASCADE=PASS, AETHER=EXPERIMENTAL_FAIL), "
        "gate, truncamento de gen_sample, cosine derivado e group_id conferidos")
    return 0


# --------------------------------- main --------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Converte compare_generations_report.json em registros "
                    "schema v2 (CMP_<TECH>_GENERATIONS) e publica no ingest "
                    "(docs/C3_CONTRACTS_V1.md §18.2)."
    )
    parser.add_argument(
        "report", nargs="?", default=DEFAULT_REPORT_PATH,
        help=f"caminho do relatório (default: {DEFAULT_REPORT_PATH})")
    parser.add_argument(
        "--results-endpoint",
        default=os.environ.get("RIFT_RESULTS_ENDPOINT") or DEFAULT_ENDPOINT,
        help="endpoint do ingest (env RIFT_RESULTS_ENDPOINT; HTTPS obrigatório)")
    parser.add_argument(
        "--publish", choices=("on", "off"), default="on",
        help="on (default): POST {records:[...]}; off: só converte/grava")
    parser.add_argument(
        "--run-id", default=None,
        help="run_id dos registros (default: cmp-<sha8 do conteúdo do relatório>)")
    parser.add_argument(
        "--selftest", action="store_true",
        help="valida a conversão com fixture embutida (sem rede/filesystem)")
    args = parser.parse_args(argv)

    if args.selftest:
        return selftest()

    report_path = os.path.abspath(os.path.expanduser(args.report))
    if not os.path.isfile(report_path):
        log(f"ERRO: relatório não encontrado: {report_path}")
        return 2
    try:
        with open(report_path, "rb") as fh:
            raw_bytes = fh.read()
        report = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        log(f"ERRO: relatório ilegível/JSON inválido: {exc}")
        return 2

    run_id = (args.run_id or "").strip() or (
        "cmp-" + hashlib.sha256(raw_bytes).hexdigest()[:8])
    try:
        records = convert_report(report, run_id, utc_now())
    except ValueError as exc:
        log(f"ERRO: {exc}")
        return 2

    records_path = os.path.join(os.path.dirname(report_path), RECORDS_BASENAME)
    try:
        with open(records_path, "w", encoding="utf-8") as fh:
            json.dump(records, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        log(f"registros convertidos gravados em {records_path}")
    except OSError as exc:
        log(f"AVISO: não consegui gravar {records_path} ({exc}) — seguindo")

    log(f"run_id={run_id} | modelo={report.get('model')} | "
        f"{len(records)} registro(s):")
    for record in records:
        log(f"  {record['battery_id']}: {record['status']}")

    if args.publish == "on":
        publish_records(records, args.results_endpoint)
    else:
        log("publicação desligada (--publish off)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
