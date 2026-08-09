"""Runtime de referência Python — CASCADE C0 (uma Linear real)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from cascade.compiler.decompose import CascadeLinearStages
from cascade.kernels.fused_stage import fused_stage_linear
from cascade.runtime.confidence_gate import GateConfig, decide_gate


@dataclass
class CascadeMetrics:
    f0_calls: int = 0
    f1_calls: int = 0
    f1_skip_rate: float = 0.0
    avg_stages_per_token: float = 1.0
    path: str = "F0_GATE_F1"


class CascadeLinearRuntime:
    """Executa Y = F0(X) + Gate(X)·F1(X) sem reconstruir W denso no caminho gated."""

    def __init__(
        self,
        stages: CascadeLinearStages,
        *,
        gate_percentile: float = 70.0,
        device: Optional[torch.device] = None,
    ):
        self.stages = stages
        self.gate_cfg = GateConfig(percentile=gate_percentile)
        self.device = device or torch.device("cpu")
        # move tensors
        self.codes = stages.codes.to(self.device)
        self.scales = stages.scales.to(self.device)
        self.u = stages.u.to(self.device)
        self.s = stages.s.to(self.device)
        self.v = stages.v.to(self.device)

    def execute(
        self,
        x: torch.Tensor,
        *,
        path: str = "F0_GATE_F1",
    ) -> Dict[str, Any]:
        """path ∈ {ORIGINAL_UNAVAILABLE, F0_ONLY, F0_PLUS_F1_ALWAYS, F0_GATE_F1}."""
        x = x.to(device=self.device, dtype=torch.float32)
        st = self.stages
        if path == "F0_ONLY":
            out = fused_stage_linear(
                x,
                codes=self.codes, scales=self.scales, group_size=st.group_size,
                out_features=st.out_features, in_features=st.in_features,
                u=None, s=None, v=None, gate_mask=None,
            )
            metrics = CascadeMetrics(
                f0_calls=int(x.shape[0]), f1_calls=0, f1_skip_rate=1.0,
                avg_stages_per_token=1.0, path=path,
            )
            return {"y": out["y"], "metrics": metrics, "gate": None}

        if path == "F0_PLUS_F1_ALWAYS":
            out = fused_stage_linear(
                x,
                codes=self.codes, scales=self.scales, group_size=st.group_size,
                out_features=st.out_features, in_features=st.in_features,
                u=self.u, s=self.s, v=self.v, gate_mask=None,
            )
            metrics = CascadeMetrics(
                f0_calls=int(x.shape[0]), f1_calls=int(x.shape[0]), f1_skip_rate=0.0,
                avg_stages_per_token=2.0, path=path,
            )
            return {"y": out["y"], "metrics": metrics, "gate": None}

        # F0_GATE_F1 — CASCADE real
        mask, gate_meta = decide_gate(x, self.gate_cfg)
        out = fused_stage_linear(
            x,
            codes=self.codes, scales=self.scales, group_size=st.group_size,
            out_features=st.out_features, in_features=st.in_features,
            u=self.u, s=self.s, v=self.v, gate_mask=mask,
        )
        f0_calls = int(x.shape[0])
        f1_calls = int(out["f1_calls"])
        skip = 1.0 - (f1_calls / max(f0_calls, 1))
        metrics = CascadeMetrics(
            f0_calls=f0_calls,
            f1_calls=f1_calls,
            f1_skip_rate=skip,
            avg_stages_per_token=1.0 + (f1_calls / max(f0_calls, 1)),
            path="F0_GATE_F1",
        )
        return {"y": out["y"], "metrics": metrics, "gate": gate_meta}
