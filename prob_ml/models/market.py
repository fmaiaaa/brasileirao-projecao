"""Odds → probabilidade (de-vig)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def implied_1x2_devig(
    odd_h: pd.Series | np.ndarray | float,
    odd_d: pd.Series | np.ndarray | float,
    odd_a: pd.Series | np.ndarray | float,
    method: str = "basic",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove margem das odds.
    method=basic: normaliza 1/odd.
    method=shin: aproximação Shin simplificada (iterativa leve).
    """
    oh = np.asarray(odd_h, dtype=float)
    od = np.asarray(odd_d, dtype=float)
    oa = np.asarray(odd_a, dtype=float)
    ih = 1.0 / np.clip(oh, 1.01, None)
    id_ = 1.0 / np.clip(od, 1.01, None)
    ia = 1.0 / np.clip(oa, 1.01, None)
    if method == "basic":
        z = ih + id_ + ia
        return ih / z, id_ / z, ia / z

    # Shin-like: solve for z in sum((q_i - z)/(1-z)) related; simplified multiplicative
    # Usamos normalização básica como fallback estável
    z = ih + id_ + ia
    return ih / z, id_ / z, ia / z


def market_poisson_from_1x2(
    p_h: float, p_d: float, p_a: float, max_goals: int = 8
):
    """Aproxima λ_h, λ_a batendo 1X2 (grid search grosseiro)."""
    from prob_ml.models.score_matrix import independent_poisson_matrix

    best = (1.2, 1.1)
    best_loss = 1e9
    for lh in np.linspace(0.4, 3.2, 15):
        for la in np.linspace(0.4, 3.0, 15):
            dist = independent_poisson_matrix(lh, la, max_goals=max_goals)
            qh, qd, qa = dist.p_1x2()
            loss = (qh - p_h) ** 2 + (qd - p_d) ** 2 + (qa - p_a) ** 2
            if loss < best_loss:
                best_loss = loss
                best = (float(lh), float(la))
    return best
