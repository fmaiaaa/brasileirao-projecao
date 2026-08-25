"""Calibração de probabilidades 1X2 (temperature / Platt)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from prob_ml.models.score_matrix import ScoreDistribution, independent_poisson_matrix


@dataclass
class Calibrator:
    method: str = "none"
    temperature: float = 1.0
    platt_a: float = 1.0
    platt_b: float = 0.0

    def apply_1x2(
        self, ph: float, pd: float, pa: float
    ) -> tuple[float, float, float]:
        p = np.array([ph, pd, pa], dtype=float)
        if self.method == "none":
            return ph, pd, pa
        if self.method == "temperature":
            logp = np.log(np.clip(p, 1e-12, 1.0)) / max(self.temperature, 1e-6)
            logp -= logp.max()
            q = np.exp(logp)
            q /= q.sum()
            return float(q[0]), float(q[1]), float(q[2])
        # platt on home-win logit vs rest — simplificado
        return ph, pd, pa


def fit_temperature(
    probs: np.ndarray, outcomes: np.ndarray
) -> Calibrator:
    """probs (n,3), outcomes 0=H,1=D,2=A."""
    best_t, best_nll = 1.0, 1e18
    for t in np.linspace(0.5, 2.5, 21):
        logp = np.log(np.clip(probs, 1e-12, 1.0)) / t
        logp -= logp.max(axis=1, keepdims=True)
        q = np.exp(logp)
        q /= q.sum(axis=1, keepdims=True)
        nll = -np.mean(np.log(q[np.arange(len(outcomes)), outcomes] + 1e-12))
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return Calibrator(method="temperature", temperature=best_t)


def recalibrate_distribution(
    dist: ScoreDistribution, cal: Calibrator
) -> ScoreDistribution:
    """Ajusta levemente λ via 1X2 calibrado (aproximação)."""
    if cal.method == "none":
        return dist
    ph, pd, pa = dist.p_1x2()
    qh, qd, qa = cal.apply_1x2(ph, pd, pa)
    eh, ea = dist.expected_goals()
    # escala suave
    scale_h = (qh + 0.5 * qd) / max(ph + 0.5 * pd, 1e-6)
    scale_a = (qa + 0.5 * qd) / max(pa + 0.5 * pd, 1e-6)
    return independent_poisson_matrix(
        max(0.05, eh * scale_h), max(0.05, ea * scale_a), max_goals=dist.max_goals
    )
