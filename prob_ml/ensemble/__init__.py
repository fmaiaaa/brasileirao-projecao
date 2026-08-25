"""Ensembles de matrizes de placar."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from prob_ml.models.score_matrix import ScoreDistribution


def blend_distributions(
    dists: Sequence[ScoreDistribution],
    weights: Sequence[float] | None = None,
) -> ScoreDistribution:
    if not dists:
        raise ValueError("lista vazia")
    w = np.asarray(weights if weights is not None else np.ones(len(dists)), dtype=float)
    w = np.clip(w, 0, None)
    if w.sum() <= 0:
        w = np.ones_like(w)
    w = w / w.sum()
    # alinha shapes
    g = max(d.max_goals for d in dists)
    mats = []
    for d in dists:
        p = d.probs
        if p.shape[0] != g + 2:  # com cauda
            # re-pad simples
            target = g + (2 if d.has_tail else 1)
            # usa probs já normalizados; se shape menor, pad
            m = np.zeros((g + 2, g + 2))
            n0, n1 = p.shape
            m[:n0, :n1] = p
            mats.append(m)
        else:
            mats.append(p)
    # reshape all to same
    shape = mats[0].shape
    mats = [m if m.shape == shape else _resize(m, shape) for m in mats]
    blended = sum(wi * mi for wi, mi in zip(w, mats))
    return ScoreDistribution(blended, max_goals=g, has_tail=True)


def _resize(m: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape)
    out[: m.shape[0], : m.shape[1]] = m
    return out


def optimize_blend_weights(
    oof_dists: list[list[ScoreDistribution]],
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    *,
    method: str = "performance_weighted",
) -> np.ndarray:
    """
    oof_dists: shape [n_models][n_matches]
    Retorna pesos >=0 somando 1.
    """
    from prob_ml.backtesting.metrics import score_nll

    n_models = len(oof_dists)
    if n_models == 0:
        return np.array([])
    nlls = np.zeros(n_models)
    for m in range(n_models):
        for j, (hg, ag) in enumerate(zip(home_goals, away_goals)):
            nlls[m] += score_nll(oof_dists[m][j], int(hg), int(ag))
    nlls /= max(len(home_goals), 1)

    if method == "simple_average":
        return np.ones(n_models) / n_models

    # performance weighted: softmax(-nll)
    logits = -nlls
    logits = logits - logits.max()
    w = np.exp(logits)
    w = w / w.sum()

    if method == "constrained_blend":
        # grid search simplex 1D para 2-4 modelos; senão usa performance_weighted
        if n_models == 2:
            best_w, best = w, 1e18
            for a in np.linspace(0, 1, 21):
                ww = np.array([a, 1 - a])
                loss = 0.0
                for j, (hg, ag) in enumerate(zip(home_goals, away_goals)):
                    d = blend_distributions(
                        [oof_dists[0][j], oof_dists[1][j]], ww
                    )
                    loss += score_nll(d, int(hg), int(ag))
                if loss < best:
                    best, best_w = loss, ww
            return best_w
    return w
