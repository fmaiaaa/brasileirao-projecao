"""Monte Carlo do campeonato a partir de matrizes de placar."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from prob_ml.models.score_matrix import ScoreDistribution, sample_score


@dataclass
class TieBreakerConfig:
    order: list[str] = field(
        default_factory=lambda: [
            "points",
            "wins",
            "goal_diff",
            "goals_for",
        ]
    )


@dataclass
class SeasonSimResult:
    teams: list[str]
    points_mean: np.ndarray
    points_median: np.ndarray
    points_p10: np.ndarray
    points_p90: np.ndarray
    position_mean: np.ndarray
    position_matrix: np.ndarray  # (n_teams, n_pos)
    p_champion: np.ndarray
    p_g4: np.ndarray
    p_g6: np.ndarray
    p_z4: np.ndarray
    n_sims: int


def _rank_rows(
    pts: np.ndarray,
    wins: np.ndarray,
    gd: np.ndarray,
    gf: np.ndarray,
) -> np.ndarray:
    """Retorna posições 1..n para um vetor de times (uma sim)."""
    order = np.lexsort((-gf, -gd, -wins, -pts))
    pos = np.empty(len(pts), dtype=int)
    pos[order] = np.arange(1, len(pts) + 1)
    return pos


def simulate_season(
    teams: Sequence[str],
    current: dict[str, dict[str, float]],
    fixtures: Sequence[tuple[str, str, ScoreDistribution]],
    *,
    n_sims: int = 20_000,
    seed: int = 42,
    tiebreakers: TieBreakerConfig | None = None,
) -> SeasonSimResult:
    """
    current[team] = {points, wins, gd, gf}
    fixtures = list of (home, away, dist)
    """
    tiebreakers = tiebreakers or TieBreakerConfig()
    teams = list(teams)
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    rng = np.random.default_rng(seed)

    pts0 = np.array([current.get(t, {}).get("points", 0.0) for t in teams])
    win0 = np.array([current.get(t, {}).get("wins", 0.0) for t in teams])
    gd0 = np.array([current.get(t, {}).get("gd", 0.0) for t in teams])
    gf0 = np.array([current.get(t, {}).get("gf", 0.0) for t in teams])

    pts = np.tile(pts0, (n_sims, 1))
    wins = np.tile(win0, (n_sims, 1))
    gd = np.tile(gd0, (n_sims, 1))
    gf = np.tile(gf0, (n_sims, 1))

    for home, away, dist in fixtures:
        ih, ia = idx[home], idx[away]
        # amostrar placares vetorizado via CDF flat
        flat = dist.probs.ravel()
        flat = flat / flat.sum()
        choices = rng.choice(flat.size, size=n_sims, p=flat)
        hs = choices // dist.probs.shape[1]
        aws = choices % dist.probs.shape[1]
        if dist.has_tail:
            hs = np.minimum(hs, dist.max_goals + 2)
            aws = np.minimum(aws, dist.max_goals + 2)

        pts[hs > aws, ih] += 3
        wins[hs > aws, ih] += 1
        pts[hs < aws, ia] += 3
        wins[hs < aws, ia] += 1
        pts[hs == aws, ih] += 1
        pts[hs == aws, ia] += 1
        gd[:, ih] += hs - aws
        gd[:, ia] += aws - hs
        gf[:, ih] += hs
        gf[:, ia] += aws

    pos_mat = np.zeros((n, n), dtype=float)
    camp = np.zeros(n)
    g4 = np.zeros(n)
    g6 = np.zeros(n)
    z4 = np.zeros(n)
    positions = np.zeros((n_sims, n), dtype=int)

    for s in range(n_sims):
        pos = _rank_rows(pts[s], wins[s], gd[s], gf[s])
        positions[s] = pos
        for i in range(n):
            pos_mat[i, pos[i] - 1] += 1
        camp[pos == 1] += 1
        g4[pos <= 4] += 1
        g6[pos <= 6] += 1
        z4[pos >= n - 3] += 1

    inv = float(n_sims)
    return SeasonSimResult(
        teams=teams,
        points_mean=pts.mean(axis=0),
        points_median=np.median(pts, axis=0),
        points_p10=np.percentile(pts, 10, axis=0),
        points_p90=np.percentile(pts, 90, axis=0),
        position_mean=positions.mean(axis=0),
        position_matrix=pos_mat / inv,
        p_champion=camp / inv,
        p_g4=g4 / inv,
        p_g6=g6 / inv,
        p_z4=z4 / inv,
        n_sims=n_sims,
    )


def result_to_frame(res: SeasonSimResult) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Time": res.teams,
            "Pts Esperados": np.round(res.points_mean, 1),
            "Pts Mediana": np.round(res.points_median, 1),
            "Pts P10": np.round(res.points_p10, 1),
            "Pts P90": np.round(res.points_p90, 1),
            "Pos Esperada": np.round(res.position_mean, 2),
            "Prob. Campeão": np.round(100 * res.p_champion, 1),
            "Prob. G4": np.round(100 * res.p_g4, 1),
            "Prob. G6": np.round(100 * res.p_g6, 1),
            "Prob. Z4": np.round(100 * res.p_z4, 1),
        }
    ).sort_values("Pos Esperada")
