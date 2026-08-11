"""Confidence Gate v0 — heurístico barato sobre ativação."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch


@dataclass
class GateConfig:
    percentile: float = 70.0  # L2 percentil → threshold
    use_rms: bool = True
    use_max_abs: bool = True
    # pesos experimentais (Fase 1)
    a_rms: float = 1.0
    b_max_abs: float = 0.25


def gate_features(x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Features baratas por linha do batch."""
    # x: (B, in)
    rms = torch.sqrt(torch.mean(x * x, dim=1) + 1e-12)
    max_abs = x.abs().amax(dim=1)
    var = torch.var(x, dim=1, unbiased=False)
    l2 = torch.linalg.vector_norm(x, dim=1) / (float(x.shape[1]) ** 0.5 + 1e-12)
    return {"rms": rms, "max_abs": max_abs, "variance": var, "l2": l2}


def gate_score(features: Dict[str, torch.Tensor], cfg: GateConfig) -> torch.Tensor:
    score = torch.zeros_like(features["l2"])
    if cfg.use_rms:
        score = score + cfg.a_rms * features["rms"]
    if cfg.use_max_abs:
        score = score + cfg.b_max_abs * features["max_abs"]
    # L2 normalizado sempre entra (estável)
    score = score + features["l2"]
    return score


def decide_gate(
    x: torch.Tensor,
    cfg: Optional[GateConfig] = None,
    *,
    threshold: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Retorna mask bool (B,) — True => executar F1.

    Se threshold None, usa percentil de `cfg.percentile` sobre o score do batch.
    """
    cfg = cfg or GateConfig()
    feats = gate_features(x)
    score = gate_score(feats, cfg)
    if threshold is None:
        pct = min(99.0, max(0.0, float(cfg.percentile)))
        threshold = torch.quantile(score.detach(), pct / 100.0)
    mask = score >= threshold
    meta = {
        "threshold": float(threshold.item() if torch.is_tensor(threshold) else threshold),
        "activation_rate": float(mask.float().mean().item()),
        "score_mean": float(score.mean().item()),
        "score_std": float(score.std(unbiased=False).item()) if score.numel() > 1 else 0.0,
        "features": {k: float(v.mean().item()) for k, v in feats.items()},
        "gate": "ACTIVATION_SCORE_PERCENTILE_V0",
        "percentile": float(cfg.percentile),
    }
    return mask, meta
