"""Confidence Gate v1 — corrige o bug de batch-1 do v0.

Bug corrigido: o v0 tirava o threshold por torch.quantile DO PROPRIO BATCH.
Em decode autorregressivo (batch=1) o threshold vira o proprio score e a
mascara e sempre True — F1 disparava em 100% dos tokens de decode.

v1: o threshold e CALIBRADO offline (percentil sobre ativacoes de calibracao)
e congelado no bundle. decide_gate exige threshold fixo para batch pequeno;
o percentil por batch continua disponivel apenas para batch >= min_batch
(prefill), que era o unico caso em que o v0 funcionava.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch


@dataclass
class GateConfig:
    percentile: float = 70.0
    use_rms: bool = True
    use_max_abs: bool = True
    a_rms: float = 1.0
    b_max_abs: float = 0.25
    fixed_threshold: Optional[float] = None
    min_batch_for_batch_percentile: int = 8


def gate_features(x: torch.Tensor) -> Dict[str, torch.Tensor]:
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
    return score + features["l2"]


class GateCalibrator:
    """Acumula scores durante a calibracao e congela o threshold.

    Uso:
        cal = GateCalibrator(cfg)
        for x in ativacoes_calibracao: cal.observe(x)
        cfg.fixed_threshold = cal.freeze()
    """

    def __init__(self, cfg: Optional[GateConfig] = None):
        self.cfg = cfg or GateConfig()
        self._scores: list[torch.Tensor] = []

    def observe(self, x: torch.Tensor) -> None:
        x2 = x.reshape(-1, x.shape[-1]).to(torch.float32)
        self._scores.append(gate_score(gate_features(x2), self.cfg).detach().cpu())

    def freeze(self) -> float:
        if not self._scores:
            raise RuntimeError("calibracao vazia — chame observe() antes de freeze()")
        alls = torch.cat(self._scores)
        pct = min(99.0, max(0.0, float(self.cfg.percentile)))
        thr = float(torch.quantile(alls, pct / 100.0).item())
        self.cfg.fixed_threshold = thr
        return thr


def decide_gate(
    x: torch.Tensor,
    cfg: Optional[GateConfig] = None,
    *,
    threshold: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """mask bool (B,) — True => executar F1. Threshold fixo tem prioridade."""
    cfg = cfg or GateConfig()
    feats = gate_features(x)
    score = gate_score(feats, cfg)
    thr = threshold if threshold is not None else cfg.fixed_threshold
    source = "fixed_calibrated"
    if thr is None:
        if x.shape[0] < cfg.min_batch_for_batch_percentile:
            # Sem calibracao, batch pequeno: percentil do batch e sem sentido
            # (com B=1 dispararia F1 sempre). Fail-safe: F0_ONLY.
            mask = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
            meta = {
                "threshold": None,
                "activation_rate": 0.0,
                "score_mean": float(score.mean().item()),
                "gate": "UNCALIBRATED_SMALL_BATCH_F0_ONLY",
                "percentile": float(cfg.percentile),
                "warning": "gate sem threshold calibrado em batch pequeno — "
                           "F1 desligado; calibre com GateCalibrator",
            }
            return mask, meta
        pct = min(99.0, max(0.0, float(cfg.percentile)))
        thr = float(torch.quantile(score.detach(), pct / 100.0).item())
        source = "batch_percentile_prefill"
    mask = score >= thr
    meta = {
        "threshold": float(thr),
        "threshold_source": source,
        "activation_rate": float(mask.float().mean().item()),
        "score_mean": float(score.mean().item()),
        "score_std": float(score.std(unbiased=False).item()) if score.numel() > 1 else 0.0,
        "gate": "ACTIVATION_SCORE_V1_CALIBRATED",
        "percentile": float(cfg.percentile),
    }
    return mask, meta
