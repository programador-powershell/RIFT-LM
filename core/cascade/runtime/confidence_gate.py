"""Confidence Gate v1 — threshold calibrado offline, com fail-safe em batch pequeno.

BUG CORRIGIDO (v0): o threshold saía de `torch.quantile` sobre o PRÓPRIO batch.
Em decode autorregressivo o batch é 1 token, então o quantil de um único
elemento É aquele elemento e a máscara virava `score >= score` — sempre True. O
gate não filtrava nada: o F1 rodava em 100% dos tokens de decode, exatamente o
custo que o gate existe para evitar. Medido no runtime v2: taxa 1.0 antes, 0.29
com threshold calibrado em p70.

v1: o threshold é calibrado UMA vez sobre ativações de calibração
(`GateCalibrator.observe/freeze`) e congelado em `GateConfig.fixed_threshold`
para gravar no bundle. Sem threshold e com batch pequeno, o gate escolhe
F0_ONLY (fail-safe) e emite `warning` na telemetria em vez de mentir uma taxa.
O percentil por batch continua válido a partir de
`min_batch_for_batch_percentile` (prefill), que era o único caso em que o v0
funcionava.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class GateConfig:
    percentile: float = 70.0  # L2 percentil → threshold
    use_rms: bool = True
    use_max_abs: bool = True
    # pesos experimentais (Fase 1)
    a_rms: float = 1.0
    b_max_abs: float = 0.25
    # Threshold congelado pela calibração; grave este valor no bundle.
    fixed_threshold: Optional[float] = None
    # Abaixo deste batch o percentil do próprio batch não tem sentido.
    min_batch_for_batch_percentile: int = 8


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


class GateCalibrator:
    """Acumula scores de ativações de calibração e congela o threshold.

        cal = GateCalibrator(cfg)
        for x in ativacoes_calibracao:
            cal.observe(x)
        cfg.fixed_threshold = cal.freeze()   # grave no bundle
    """

    def __init__(self, cfg: Optional[GateConfig] = None):
        self.cfg = cfg or GateConfig()
        self._scores: List[torch.Tensor] = []

    def observe(self, x: torch.Tensor) -> None:
        x2 = x.reshape(-1, x.shape[-1]).to(torch.float32)
        self._scores.append(gate_score(gate_features(x2), self.cfg).detach().cpu())

    @property
    def observed_rows(self) -> int:
        return int(sum(int(s.numel()) for s in self._scores))

    def freeze(self) -> float:
        if not self._scores:
            raise RuntimeError("calibração vazia — chame observe() antes de freeze()")
        alls = torch.cat(self._scores)
        pct = min(99.0, max(0.0, float(self.cfg.percentile)))
        thr = float(torch.quantile(alls, pct / 100.0).item())
        self.cfg.fixed_threshold = thr
        return thr


def decide_gate(
    x: torch.Tensor,
    cfg: Optional[GateConfig] = None,
    *,
    threshold: Optional[Any] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Retorna mask bool (B,) — True => executar F1.

    Precedência do threshold: argumento explícito → `cfg.fixed_threshold` →
    percentil do batch (somente com batch >= min_batch_for_batch_percentile).
    Sem nenhum dos três, o gate desliga o F1 (fail-safe) e avisa.
    """
    cfg = cfg or GateConfig()
    feats = gate_features(x)
    score = gate_score(feats, cfg)

    thr: Optional[float] = None
    if threshold is not None:
        thr = float(threshold.item() if torch.is_tensor(threshold) else threshold)
        source = "explicit_argument"
    elif cfg.fixed_threshold is not None:
        thr = float(cfg.fixed_threshold)
        source = "fixed_calibrated"
    elif int(x.shape[0]) >= int(cfg.min_batch_for_batch_percentile):
        pct = min(99.0, max(0.0, float(cfg.percentile)))
        thr = float(torch.quantile(score.detach(), pct / 100.0).item())
        source = "batch_percentile_prefill"
    else:
        # Fail-safe: sem calibração e batch pequeno (decode). Reportar uma taxa
        # aqui seria inventar um número — o v0 reportava 1.0 achando que filtrava.
        base = {k: float(v.mean().item()) for k, v in feats.items()}
        return (
            torch.zeros(int(x.shape[0]), dtype=torch.bool, device=x.device),
            {
                "threshold": None,
                "threshold_source": "uncalibrated_small_batch",
                "activation_rate": 0.0,
                "score_mean": float(score.mean().item()),
                "score_std": (
                    float(score.std(unbiased=False).item()) if score.numel() > 1 else 0.0
                ),
                "features": base,
                "gate": "UNCALIBRATED_SMALL_BATCH_F0_ONLY",
                "percentile": float(cfg.percentile),
                "warning": (
                    "gate sem threshold calibrado em batch "
                    f"{int(x.shape[0])} < {int(cfg.min_batch_for_batch_percentile)}: "
                    "F1 desligado. Calibre com GateCalibrator e grave "
                    "fixed_threshold no bundle."
                ),
            },
        )

    mask = score >= thr
    meta = {
        "threshold": float(thr),
        "threshold_source": source,
        "activation_rate": float(mask.float().mean().item()),
        "score_mean": float(score.mean().item()),
        "score_std": float(score.std(unbiased=False).item()) if score.numel() > 1 else 0.0,
        "features": {k: float(v.mean().item()) for k, v in feats.items()},
        "gate": "ACTIVATION_SCORE_V1_CALIBRATED",
        "percentile": float(cfg.percentile),
    }
    return mask, meta
