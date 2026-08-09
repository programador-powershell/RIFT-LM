"""Smoke test CASCADE-C0 without HF model."""
import torch
from cascade.compiler.decompose import decompose_linear_int4_lowrank
from cascade.compiler.bundle_writer import write_cascade_bundle
from cascade.runtime.reference import CascadeLinearRuntime
from cascade.kernels.fused_stage import fused_stage_linear
import torch.nn.functional as F
from pathlib import Path

def test_c0_synthetic(tmp_path=None):
    torch.manual_seed(0)
    w = torch.randn(128, 64)
    x = torch.randn(32, 64)
    stages = decompose_linear_int4_lowrank(w, rank=8, group_size=32)
    assert stages.f0_bytes > 0 and stages.f1_bytes > 0
    assert stages.f0_bytes + stages.f1_bytes < stages.baseline_bytes
    rt = CascadeLinearRuntime(stages, gate_percentile=50.0)
    y_ref = F.linear(x, w)
    r0 = rt.execute(x, path="F0_ONLY")
    r1 = rt.execute(x, path="F0_PLUS_F1_ALWAYS")
    rg = rt.execute(x, path="F0_GATE_F1")
    assert r0["y"].shape == y_ref.shape
    assert rg["metrics"].f1_skip_rate >= 0.0
    out = Path("/tmp/cascade_c0_smoke.cascade")
    meta = write_cascade_bundle(out, stages=stages, model_id="smoke/test", target_layer="l.weight")
    assert out.is_file() and meta["file_size"] > 128
    print("SMOKE OK", stages.to_meta(), "skip", rg["metrics"].f1_skip_rate, "bundle", meta["file_size"])

if __name__ == "__main__":
    test_c0_synthetic()
