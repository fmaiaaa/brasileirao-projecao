"""Modelos Poisson / Dixon-Coles / baselines."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from prob_ml.models.base import ScoreModel
from prob_ml.models.score_matrix import (
    ScoreDistribution,
    dixon_coles_matrix,
    independent_poisson_matrix,
)


class LeagueMeanModel(ScoreModel):
    name = "league_mean"

    def __init__(self, max_goals: int = 8):
        self.max_goals = max_goals
        self.lam_h = 1.3
        self.lam_a = 1.1

    def fit(self, matches: pd.DataFrame, features: pd.DataFrame | None = None) -> "LeagueMeanModel":
        played = matches.dropna(subset=["home_goals", "away_goals"])
        if len(played):
            self.lam_h = float(played["home_goals"].mean())
            self.lam_a = float(played["away_goals"].mean())
        return self

    def predict_match(self, row, features_row=None) -> ScoreDistribution:
        return independent_poisson_matrix(self.lam_h, self.lam_a, self.max_goals)


class IndependentPoissonAD(ScoreModel):
    """Attack/Defense ratings Poisson (máxima verossimilhança aproximada)."""

    name = "independent_poisson"

    def __init__(self, max_goals: int = 8, home_advantage: float = 0.25):
        self.max_goals = max_goals
        self.home_advantage = home_advantage
        self.attack: dict[str, float] = {}
        self.defense: dict[str, float] = {}
        self.mu = 1.2

    def fit(self, matches: pd.DataFrame, features: pd.DataFrame | None = None) -> "IndependentPoissonAD":
        played = matches.dropna(subset=["home_goals", "away_goals"]).copy()
        teams = sorted(set(played["home_team"]) | set(played["away_team"]))
        self.attack = {t: 1.0 for t in teams}
        self.defense = {t: 1.0 for t in teams}
        if played.empty:
            return self
        self.mu = float((played["home_goals"].mean() + played["away_goals"].mean()) / 2)

        # Iteração tipo Maher
        for _ in range(40):
            gf = defaultdict(float)
            ga = defaultdict(float)
            n = defaultdict(float)
            for _, r in played.iterrows():
                h, a = r["home_team"], r["away_team"]
                gf[h] += float(r["home_goals"])
                ga[h] += float(r["away_goals"])
                gf[a] += float(r["away_goals"])
                ga[a] += float(r["home_goals"])
                n[h] += 1
                n[a] += 1
            for t in teams:
                opp_def = np.mean([self.defense[x] for x in teams if x != t]) or 1.0
                opp_att = np.mean([self.attack[x] for x in teams if x != t]) or 1.0
                self.attack[t] = (gf[t] / max(n[t], 1)) / max(self.mu * opp_def, 1e-6)
                self.defense[t] = (ga[t] / max(n[t], 1)) / max(self.mu * opp_att, 1e-6)
            # normaliza
            mean_att = np.mean(list(self.attack.values()))
            mean_def = np.mean(list(self.defense.values()))
            for t in teams:
                self.attack[t] /= mean_att
                self.defense[t] /= mean_def
        return self

    def _lams(self, home: str, away: str) -> tuple[float, float]:
        ah = self.attack.get(home, 1.0)
        dh = self.defense.get(home, 1.0)
        aa = self.attack.get(away, 1.0)
        da = self.defense.get(away, 1.0)
        lh = self.mu * ah * da * (1.0 + self.home_advantage)
        la = self.mu * aa * dh
        return max(0.05, lh), max(0.05, la)

    def predict_match(self, row, features_row=None) -> ScoreDistribution:
        lh, la = self._lams(str(row["home_team"]), str(row["away_team"]))
        return independent_poisson_matrix(lh, la, self.max_goals)


class DixonColesModel(IndependentPoissonAD):
    name = "dixon_coles"

    def __init__(self, max_goals: int = 8, home_advantage: float = 0.25, rho: float = -0.08):
        super().__init__(max_goals=max_goals, home_advantage=home_advantage)
        self.rho = rho

    def fit(self, matches: pd.DataFrame, features: pd.DataFrame | None = None) -> "DixonColesModel":
        super().fit(matches, features)
        # estima rho simples via NLL em grade
        played = matches.dropna(subset=["home_goals", "away_goals"])
        best_rho, best_nll = self.rho, 1e18
        for rho in np.linspace(-0.2, 0.1, 13):
            nll = 0.0
            for _, r in played.iterrows():
                lh, la = self._lams(str(r["home_team"]), str(r["away_team"]))
                dist = dixon_coles_matrix(lh, la, rho=float(rho), max_goals=self.max_goals)
                i = min(int(r["home_goals"]), self.max_goals)
                j = min(int(r["away_goals"]), self.max_goals)
                nll -= np.log(max(dist.probs[i, j], 1e-12))
            if nll < best_nll:
                best_nll, best_rho = nll, float(rho)
        self.rho = best_rho
        return self

    def predict_match(self, row, features_row=None) -> ScoreDistribution:
        lh, la = self._lams(str(row["home_team"]), str(row["away_team"]))
        return dixon_coles_matrix(lh, la, rho=self.rho, max_goals=self.max_goals)


class EloScoreModel(ScoreModel):
    name = "elo_result"

    def __init__(self, max_goals: int = 8, k: float = 20.0, home_adv: float = 60.0):
        self.max_goals = max_goals
        self.k = k
        self.home_adv = home_adv
        self.ratings: dict[str, float] = {}
        self.league_home = 1.3
        self.league_away = 1.1

    def fit(self, matches: pd.DataFrame, features: pd.DataFrame | None = None) -> "EloScoreModel":
        from prob_ml.ratings import update_elo

        out, state = update_elo(matches, k=self.k, home_adv=self.home_adv)
        self.ratings = dict(state.ratings)
        played = matches.dropna(subset=["home_goals", "away_goals"])
        if len(played):
            self.league_home = float(played["home_goals"].mean())
            self.league_away = float(played["away_goals"].mean())
        self._pre = out
        return self

    def predict_match(self, row, features_row=None) -> ScoreDistribution:
        from prob_ml.ratings import expected_score
        from prob_ml.models.market import market_poisson_from_1x2

        h = str(row["home_team"])
        a = str(row["away_team"])
        rh = self.ratings.get(h, 1500.0)
        ra = self.ratings.get(a, 1500.0)
        ph = expected_score(rh + self.home_adv, ra)
        # empate tipicamente ~0.28
        pd_ = 0.28
        pa = max(1e-6, 1.0 - ph)
        # renormaliza com draw
        scale = ph + pa
        ph, pa = ph / scale * (1 - pd_), pa / scale * (1 - pd_)
        lh, la = market_poisson_from_1x2(ph, pd_, pa, max_goals=self.max_goals)
        # mistura com média da liga
        lh = 0.7 * lh + 0.3 * self.league_home
        la = 0.7 * la + 0.3 * self.league_away
        return independent_poisson_matrix(lh, la, self.max_goals)


class PoissonGLMModel(ScoreModel):
    """GLM Poisson com features pré-jogo (sklearn PoissonRegressor)."""

    name = "poisson_glm"

    def __init__(self, max_goals: int = 8, alpha: float = 1.0):
        self.max_goals = max_goals
        self.alpha = alpha
        self.model_h = None
        self.model_a = None
        self.feature_cols: list[str] = []
        self.fallback = LeagueMeanModel(max_goals)

    def _select_cols(self, features: pd.DataFrame) -> list[str]:
        cols = [
            c
            for c in features.columns
            if c.startswith(("home_gf_", "home_ga_", "away_gf_", "away_ga_", "home_pts_", "away_pts_", "elo_", "mkt_", "league_"))
            or c.endswith(("_roll3", "_roll5", "_roll8", "_ewm3", "_ewm5"))
        ]
        # prefer numeric
        out = []
        for c in cols:
            if pd.api.types.is_numeric_dtype(features[c]):
                out.append(c)
        return out[:40]

    def fit(self, matches: pd.DataFrame, features: pd.DataFrame | None = None) -> "PoissonGLMModel":
        self.fallback.fit(matches, features)
        if features is None:
            return self
        try:
            from sklearn.linear_model import PoissonRegressor
        except ImportError:
            return self

        played = matches["home_goals"].notna() & matches["away_goals"].notna()
        self.feature_cols = self._select_cols(features)
        if not self.feature_cols:
            return self
        X = features.loc[played, self.feature_cols].fillna(0.0).to_numpy()
        yh = matches.loc[played, "home_goals"].to_numpy(dtype=float)
        ya = matches.loc[played, "away_goals"].to_numpy(dtype=float)
        if len(yh) < 20:
            return self
        self.model_h = PoissonRegressor(alpha=self.alpha, max_iter=300)
        self.model_a = PoissonRegressor(alpha=self.alpha, max_iter=300)
        self.model_h.fit(X, yh)
        self.model_a.fit(X, ya)
        return self

    def predict_match(self, row, features_row=None) -> ScoreDistribution:
        if self.model_h is None or features_row is None or not self.feature_cols:
            return self.fallback.predict_match(row, features_row)
        x = np.asarray(
            features_row.reindex(self.feature_cols).to_numpy(dtype=float),
            dtype=float,
        )
        x = np.nan_to_num(x, nan=0.0).reshape(1, -1)
        lh = float(self.model_h.predict(x)[0])
        la = float(self.model_a.predict(x)[0])
        return independent_poisson_matrix(lh, la, self.max_goals)


class ElasticNetGoalsModel(ScoreModel):
    name = "elastic_net_goals"

    def __init__(self, max_goals: int = 8, alpha: float = 0.2, l1_ratio: float = 0.5):
        self.max_goals = max_goals
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.model_h = None
        self.model_a = None
        self.feature_cols: list[str] = []
        self.fallback = LeagueMeanModel(max_goals)

    def fit(self, matches: pd.DataFrame, features: pd.DataFrame | None = None) -> "ElasticNetGoalsModel":
        self.fallback.fit(matches, features)
        if features is None:
            return self
        try:
            from sklearn.linear_model import ElasticNet
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
        except ImportError:
            return self

        played = matches["home_goals"].notna() & matches["away_goals"].notna()
        cols = [
            c
            for c in features.columns
            if pd.api.types.is_numeric_dtype(features[c])
            and (
                "roll" in c
                or "ewm" in c
                or c.startswith("mkt_")
                or c.startswith("elo_")
                or c == "league_avg_goals_pre"
            )
        ]
        self.feature_cols = cols[:50]
        if len(self.feature_cols) < 3 or played.sum() < 25:
            return self
        X = features.loc[played, self.feature_cols].fillna(0.0)
        yh = matches.loc[played, "home_goals"].astype(float)
        ya = matches.loc[played, "away_goals"].astype(float)
        self.model_h = Pipeline(
            [
                ("sc", StandardScaler()),
                ("en", ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=4000)),
            ]
        )
        self.model_a = Pipeline(
            [
                ("sc", StandardScaler()),
                ("en", ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=4000)),
            ]
        )
        self.model_h.fit(X, yh)
        self.model_a.fit(X, ya)
        return self

    def predict_match(self, row, features_row=None) -> ScoreDistribution:
        if self.model_h is None or features_row is None:
            return self.fallback.predict_match(row, features_row)
        x = features_row.reindex(self.feature_cols).to_frame().T
        x = x.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        lh = max(0.05, float(self.model_h.predict(x)[0]))
        la = max(0.05, float(self.model_a.predict(x)[0]))
        return independent_poisson_matrix(lh, la, self.max_goals)


def build_model_zoo(max_goals: int = 8, enabled: list[str] | None = None) -> list[ScoreModel]:
    catalog: dict[str, ScoreModel] = {
        "league_mean": LeagueMeanModel(max_goals),
        "elo_result": EloScoreModel(max_goals),
        "independent_poisson": IndependentPoissonAD(max_goals),
        "dixon_coles": DixonColesModel(max_goals),
        "poisson_glm": PoissonGLMModel(max_goals),
        "elastic_net_goals": ElasticNetGoalsModel(max_goals),
    }
    if enabled is None:
        return list(catalog.values())
    return [catalog[n] for n in enabled if n in catalog]
