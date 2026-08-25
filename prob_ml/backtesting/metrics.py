"""Métricas probabilísticas."""
from __future__ import annotations

import numpy as np

from prob_ml.models.score_matrix import ScoreDistribution


def score_nll(dist: ScoreDistribution, hg: int, ag: int) -> float:
    i = min(int(hg), dist.max_goals)
    j = min(int(ag), dist.max_goals)
    if dist.has_tail and (int(hg) > dist.max_goals or int(ag) > dist.max_goals):
        # usa cauda
        i = dist.probs.shape[0] - 1 if int(hg) > dist.max_goals else i
        j = dist.probs.shape[1] - 1 if int(ag) > dist.max_goals else j
    p = float(dist.probs[i, j])
    return -np.log(max(p, 1e-12))


def logloss_1x2(dist: ScoreDistribution, hg: int, ag: int) -> float:
    ph, pd, pa = dist.p_1x2()
    if hg > ag:
        p = ph
    elif hg < ag:
        p = pa
    else:
        p = pd
    return -np.log(max(p, 1e-12))


def brier_1x2(dist: ScoreDistribution, hg: int, ag: int) -> float:
    ph, pd, pa = dist.p_1x2()
    y = np.array([hg > ag, hg == ag, hg < ag], dtype=float)
    p = np.array([ph, pd, pa])
    return float(np.sum((p - y) ** 2))


def rps_1x2(dist: ScoreDistribution, hg: int, ag: int) -> float:
    ph, pd, pa = dist.p_1x2()
    # ordem Home, Draw, Away
    p = np.array([ph, pd, pa])
    if hg > ag:
        y = np.array([1.0, 0.0, 0.0])
    elif hg == ag:
        y = np.array([0.0, 1.0, 0.0])
    else:
        y = np.array([0.0, 0.0, 1.0])
    cp = np.cumsum(p)
    cy = np.cumsum(y)
    return float(np.sum((cp - cy) ** 2) / (len(p) - 1))


def aggregate_metrics(
    dists: list[ScoreDistribution],
    home_goals: np.ndarray,
    away_goals: np.ndarray,
) -> dict[str, float]:
    n = len(dists)
    if n == 0:
        return {
            "score_nll": float("nan"),
            "ll_1x2": float("nan"),
            "brier": float("nan"),
            "rps": float("nan"),
            "n": 0,
        }
    sn = ll = br = rp = 0.0
    for d, hg, ag in zip(dists, home_goals, away_goals):
        sn += score_nll(d, int(hg), int(ag))
        ll += logloss_1x2(d, int(hg), int(ag))
        br += brier_1x2(d, int(hg), int(ag))
        rp += rps_1x2(d, int(hg), int(ag))
    return {
        "score_nll": sn / n,
        "ll_1x2": ll / n,
        "brier": br / n,
        "rps": rp / n,
        "n": n,
    }
