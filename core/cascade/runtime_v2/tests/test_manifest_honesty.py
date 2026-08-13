#!/usr/bin/env python3
"""Contrato de honestidade do manifest (§33/§34) — executavel.

§33: o docstring do conversor aponta LADDER como fonte da verdade e nao
     promete degraus/classes que o codigo nao tem.
§34: o resumo do manifest expoe all_tensors_passed_gate,
     below_gate_tensor_count e below_gate_tensors; o console imprime ATENCAO
     quando algo grava via RESCUE_LAST_RUNG; a residencia carrega
     regra_orcamento e folga_gib por classe.

Roda duas conversoes reais em miniatura: uma limpa (tudo passa) e uma
adversarial (tensor com spikes que reprova o gate)."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG))
from cascade_runtime_v2 import convert as cv


def _run_convert(inp: Path, out: Path) -> str:
    r = subprocess.run(
        [sys.executable, "-m", "cascade_runtime_v2.convert",
         "--input", str(inp), "--output", str(out), "--model-id", "teste"],
        cwd=str(PKG), capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout


def main() -> int:
    # ---- §33: docstring vs codigo -------------------------------------
    doc = cv.__doc__ or ""
    assert "FONTE DA VERDADE" in doc and "LADDER" in doc
    assert "8/16/24/40" not in doc, "classes antigas no docstring"
    assert cv.LADDER == (("q4k", 64, 4), ("q4k", 32, 4)), cv.LADDER
    assert "q5k" not in doc.split("pendencia")[0].split("FONTE")[0], \
        "docstring promete degrau inexistente antes de declarar pendencia"
    assert cv.MACHINE_TOTAL_GIB == (16, 24, 32, 48)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        torch.manual_seed(0)

        # ---- caso limpo: tudo passa o gate ----------------------------
        clean = td / "clean"
        clean.mkdir()
        save_file({"layers.0.a.weight": torch.randn(256, 512) * 0.02,
                   "norm.weight": torch.ones(256)},
                  str(clean / "model.safetensors"))
        out1 = _run_convert(clean, td / "out1")
        man1 = json.loads((td / "out1" / "cascade_manifest.json").read_text())
        s1 = man1["summary"]
        for campo in ("all_tensors_passed_gate", "below_gate_tensor_count",
                      "below_gate_tensors"):
            assert campo in s1, f"§34: {campo} ausente do resumo"
        assert s1["all_tensors_passed_gate"] is True
        assert s1["below_gate_tensor_count"] == 0
        assert "ATENCAO" not in out1
        res = s1["residencia"]
        assert "regra_orcamento" in res and "total - 8 GiB" in res["regra_orcamento"]
        for cls in ("maquina_16gb", "maquina_24gb", "maquina_32gb", "maquina_48gb"):
            c = res["classes"][cls]
            assert set(c) == {"orcamento_gib", "cabe", "folga_gib"}, c
        assert res["classes"]["maquina_24gb"]["orcamento_gib"] == 16.0, \
            "anti-regressao §28: 24 GB -> orcamento 16 GiB"

        # ---- caso adversarial: forca RESCUE_LAST_RUNG ------------------
        for amp in (5.0, 25.0, 125.0):
            # dois outliers opostos POR GRUPO de 32: estica o range do grupo
            # e esmaga os 30 valores pequenos -> ~2% da energia destruida
            w = torch.randn(256, 512) * 0.001
            w[:, 0::32] = amp
            w[:, 1::32] = -amp
            adv = td / f"adv{int(amp)}"
            adv.mkdir()
            save_file({"layers.0.b.weight": w},
                      str(adv / "model.safetensors"))
            out2 = _run_convert(adv, td / f"o{int(amp)}")
            man2 = json.loads(
                (td / f"o{int(amp)}" / "cascade_manifest.json").read_text())
            s2 = man2["summary"]
            if not s2["all_tensors_passed_gate"]:
                break
        else:
            raise AssertionError("nao consegui construir tensor abaixo do gate")
        assert s2["below_gate_tensor_count"] == 1
        assert s2["below_gate_tensors"] == ["layers.0.b.weight"]
        assert "ATENCAO" in out2, "console nao avisou RESCUE_LAST_RUNG"
        rec = [t for t in man2["tensors"] if t["name"] == "layers.0.b.weight"][0]
        assert rec.get("quality_flag") == "abaixo_do_gate_verificar_e2e"
        assert rec["ladder"][-1].get("gate") == "RESCUE_LAST_RUNG"

    print("test_manifest_honesty: contrato §33/§34 PASS "
          "(limpo aprovado-por-construcao; adversarial flagrado no resumo + console)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
