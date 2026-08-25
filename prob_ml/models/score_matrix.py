"""Utilitários de matriz de placar e derivadas."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import poisson


@dataclass
class ScoreDistribution:
    """P[i, j] = P(home=i, away=j), i,j em 0..max_goals (+ cauda opcional)."""

    probs: np.ndarray  # shape (G+1, G+1) ou (G+2,G+2) com cauda
    max_goals: int
    has_tail: bool = False

    def __post_init__(self) -> None:
        self.probs = np.asarray(self.probs, dtype=float)
        self.normalize()

    def normalize(self) -> None:
        s = float(self.probs.sum())
        if s <= 0:
            g = self.probs.shape[0]
            self.probs = np.full((g, g), 1.0 / (g * g))
        else:
            self.probs = self.probs / s

    def p_1x2(self) -> tuple[float, float, float]:
        p = self.probs
        n = self.max_goals + 1
        home = float(np.tril(p[:n, :n], -1).sum())  # i > j
        # wait: home wins when i > j → lower triangle excluding diag... tril with -1 is i>j if rows=i cols=j? 
        # In numpy tril(m, -1): elements on and below diagonal... for row i col j, tril keeps i >= j.
        # Home win: home_goals > away_goals → i > j → below diagonal in standard (row=home, col=away) is i > j which is tril -1? 
        # tril(k=-1) keeps i-j >= 1 i.e. i >= j+1 i.e. i > j. Yes.
        draw = float(np.trace(p[:n, :n]))
        away = float(np.triu(p[:n, :n], 1).sum())
        if self.has_tail:
            # cauda: última linha/coluna — aproximação grosseira
            home += float(p[n:, :n].sum())  # home many goals
            away += float(p[:n, n:].sum())
            # canto cauda-cauda: split 1/3
            corner = float(p[n:, n:].sum())
            home += corner / 3
            draw += corner / 3
            away += corner / 3
        z = home + draw + away
        if z <= 0:
            return 1 / 3, 1 / 3, 1 / 3
        return home / z, draw / z, away / z

    def expected_goals(self) -> tuple[float, float]:
        g = np.arange(self.probs.shape[0], dtype=float)
        if self.has_tail:
            g = g.copy()
            g[-1] = self.max_goals + 1.5  # proxy cauda
        eh = float((self.probs.sum(axis=1) * g).sum())
        ea = float((self.probs.sum(axis=0) * g).sum())
        return eh, ea

    def expected_points(self) -> tuple[float, float]:
        ph, pd, pa = self.p_1x2()
        return 3 * ph + pd, 3 * pa + pd

    def top_scores(self, k: int = 5) -> list[tuple[int, int, float]]:
        p = self.probs
        n = self.max_goals + 1
        flat = [
            (i, j, float(p[i, j]))
            for i in range(n)
            for j in range(n)
        ]
        flat.sort(key=lambda x: -x[2])
        return flat[:k]

    def over_under(self, line: float = 2.5) -> tuple[float, float]:
        n = self.max_goals + 1
        over = 0.0
        under = 0.0
        for i in range(n):
            for j in range(n):
                tot = i + j
                if tot > line:
                    over += self.probs[i, j]
                elif tot < line:
                    under += self.probs[i, j]
                else:
                    over += self.probs[i, j] * 0.5
                    under += self.probs[i, j] * 0.5
        z = over + under
        return (over / z, under / z) if z else (0.5, 0.5)

    def btts(self) -> tuple[float, float]:
        n = self.max_goals + 1
        yes = float(self.probs[1:n, 1:n].sum())
        no = float(self.probs[:n, :n].sum() - yes)
        z = yes + no
        return (yes / z, no / z) if z else (0.5, 0.5)


def independent_poisson_matrix(
    lam_h: float, lam_a: float, max_goals: int = 8
) -> ScoreDistribution:
    lam_h = max(1e-6, float(lam_h))
    lam_a = max(1e-6, float(lam_a))
    xs = np.arange(0, max_goals + 1)
    ph = poisson.pmf(xs, lam_h)
    pa = poisson.pmf(xs, lam_a)
    # massa na cauda (>max_goals)
    th = float(1.0 - ph.sum())
    ta = float(1.0 - pa.sum())
    ph = np.append(ph, max(th, 0.0))
    pa = np.append(pa, max(ta, 0.0))
    mat = np.outer(ph, pa)
    return ScoreDistribution(mat, max_goals=max_goals, has_tail=True)


def dixon_coles_matrix(
    lam_h: float,
    lam_a: float,
    rho: float = -0.1,
    max_goals: int = 8,
) -> ScoreDistribution:
    """Dixon-Coles com correção tau em 0-0,1-0,0-1,1-1."""
    base = independent_poisson_matrix(lam_h, lam_a, max_goals=max_goals)
    p = base.probs.copy()
    lh, la = max(1e-6, lam_h), max(1e-6, lam_a)

    def tau(i: int, j: int) -> float:
        if i == 0 and j == 0:
            return 1.0 - lh * la * rho
        if i == 0 and j == 1:
            return 1.0 + lh * rho
        if i == 1 and j == 0:
            return 1.0 + la * rho
        if i == 1 and j == 1:
            return 1.0 - rho
        return 1.0

    for i in range(min(2, p.shape[0])):
        for j in range(min(2, p.shape[1])):
            p[i, j] *= tau(i, j)
    return ScoreDistribution(p, max_goals=max_goals, has_tail=base.has_tail)


def sample_score(
    dist: ScoreDistribution, rng: np.random.Generator
) -> tuple[int, int]:
    flat = dist.probs.ravel()
    idx = int(rng.choice(flat.size, p=flat / flat.sum()))
    i, j = divmod(idx, dist.probs.shape[1])
    if dist.has_tail and i > dist.max_goals:
        i = dist.max_goals + int(rng.poisson(0.5)) + 1
    if dist.has_tail and j > dist.max_goals:
        j = dist.max_goals + int(rng.poisson(0.5)) + 1
    return int(i), int(j)


def assert_valid_distribution(dist: ScoreDistribution, tol: float = 1e-6) -> None:
    assert np.all(dist.probs >= -tol)
    assert abs(dist.probs.sum() - 1.0) < 1e-5
    ph, pd, pa = dist.p_1x2()
    assert abs(ph + pd + pa - 1.0) < 1e-5
