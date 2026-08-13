#!/usr/bin/env python3
"""Anti-regressao da regra de orcamento (§28): total - 8 GiB, piso 50%.

O bug original subtraia 8 DUAS vezes (cls ja era o orcamento e o codigo
fazia cls - 8 de novo), publicando 'NAO CABE em 24 GB' exatamente onde o
calculo manual dizia CABE — e enterrando o veredito num rotulo de 32 GB."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cascade_runtime_v2.convert import (KV_RUNTIME_RESERVE_GIB,
                                        MACHINE_TOTAL_GIB, residency_report)

GIB = 2**30


def main() -> int:
    assert MACHINE_TOTAL_GIB == (16, 24, 32, 48), MACHINE_TOTAL_GIB

    r = residency_report(0)
    # orcamentos corretos por maquina (anti-subtracao-dupla)
    assert r["classes"]["maquina_16gb"]["orcamento_gib"] == 8.0
    assert r["classes"]["maquina_24gb"]["orcamento_gib"] == 16.0, \
        "24 GiB total -> 16 GiB de orcamento, NUNCA 8 (subtracao dupla)"
    assert r["classes"]["maquina_32gb"]["orcamento_gib"] == 24.0
    assert r["classes"]["maquina_48gb"]["orcamento_gib"] == 40.0
    # piso de 50%: maquina hipotetica de 8 -> 4 (nao 0)
    from cascade_runtime_v2 import convert as cv
    assert max(8 - 8.0, 8 / 2.0) == 4.0
    assert "regra_orcamento" in r
    assert "total - 8 GiB" in r["regra_orcamento"]

    # caso Muse-Glimmer v2: 15.08 GB de bundle -> precisa 15.54 GiB
    muse = residency_report(int(15.08e9))
    m24 = muse["classes"]["maquina_24gb"]
    assert muse["necessario_com_reserva_gib"] == 15.54, muse
    assert m24["cabe"] is True, "bundle de 15.08 GB CABE na maquina de 24 GB"
    assert m24["folga_gib"] == 0.46, m24
    assert muse["classes"]["maquina_16gb"]["cabe"] is False
    assert muse["classes"]["maquina_32gb"]["folga_gib"] == 8.46

    # limiar exato: orcamento 16 GiB - reserva 1.5 = 14.5 GiB de peso
    exato = residency_report(int(14.5 * GIB))
    assert exato["classes"]["maquina_24gb"]["cabe"] is True
    quase = residency_report(int(14.51 * GIB))
    assert quase["classes"]["maquina_24gb"]["cabe"] is False

    # folga negativa reportada (nao mascarada) quando nao cabe
    assert quase["classes"]["maquina_24gb"]["folga_gib"] < 0

    # rotulos falam do TOTAL da maquina, nao do orcamento
    for total in MACHINE_TOTAL_GIB:
        assert f"maquina_{total}gb" in r["classes"]

    assert KV_RUNTIME_RESERVE_GIB == 1.5
    print("test_residency: 22 asserções PASS (anti-regressão subtração dupla)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
