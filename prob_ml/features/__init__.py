"""Feature registry + engenharia leakage-safe (shift → rolling)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass
class FeatureMeta:
    feature_name: str
    source: str
    group: str
    window: int | None = None
    transformation: str = "raw"
    available_at: str = "pre_kickoff"
    leakage_safe: bool = True
    home_away_specific: bool = False
    opponent_adjusted: bool = False
    normalized: bool = False
    version: str = "1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FeatureRegistry:
    def __init__(self) -> None:
        self._items: dict[str, FeatureMeta] = {}

    def register(self, meta: FeatureMeta) -> None:
        self._items[meta.feature_name] = meta

    def get(self, name: str) -> FeatureMeta | None:
        return self._items.get(name)

    def list(self, *, group: str | None = None) -> list[FeatureMeta]:
        vals = list(self._items.values())
        if group:
            vals = [m for m in vals if m.group == group]
        return vals

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame([m.to_dict() for m in self._items.values()])


def _team_long(df: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por time-partida com métricas do ponto de vista do time."""
    home = pd.DataFrame(
        {
            "date": df["date"],
            "season": df.get("season"),
            "round": df.get("round"),
            "team": df["home_team"],
            "opp": df["away_team"],
            "is_home": 1,
            "gf": df["home_goals"],
            "ga": df["away_goals"],
            "xg_for": df["home_xg"] if "home_xg" in df.columns else np.nan,
            "xg_against": df["away_xg"] if "away_xg" in df.columns else np.nan,
            "shots_for": df["home_shots"] if "home_shots" in df.columns else np.nan,
            "shots_against": df["away_shots"] if "away_shots" in df.columns else np.nan,
            "sot_for": df["home_sot"] if "home_sot" in df.columns else np.nan,
            "sot_against": df["away_sot"] if "away_sot" in df.columns else np.nan,
        }
    )
    away = pd.DataFrame(
        {
            "date": df["date"],
            "season": df.get("season"),
            "round": df.get("round"),
            "team": df["away_team"],
            "opp": df["home_team"],
            "is_home": 0,
            "gf": df["away_goals"],
            "ga": df["home_goals"],
            "xg_for": df["away_xg"] if "away_xg" in df.columns else np.nan,
            "xg_against": df["home_xg"] if "home_xg" in df.columns else np.nan,
            "shots_for": df["away_shots"] if "away_shots" in df.columns else np.nan,
            "shots_against": df["home_shots"] if "home_shots" in df.columns else np.nan,
            "sot_for": df["away_sot"] if "away_sot" in df.columns else np.nan,
            "sot_against": df["home_sot"] if "home_sot" in df.columns else np.nan,
        }
    )
    long = pd.concat([home, away], ignore_index=True)
    long["pts"] = np.where(
        long["gf"] > long["ga"], 3, np.where(long["gf"] == long["ga"], 1, 0)
    )
    long["gd"] = long["gf"] - long["ga"]
    long = long.sort_values(["team", "date"], kind="mergesort").reset_index(drop=True)
    return long


def _rolling_shifted(
    s: pd.Series, window: int, min_periods: int = 1
) -> pd.Series:
    """shift(1) → rolling: nunca inclui o jogo atual."""
    return s.shift(1).rolling(window, min_periods=min_periods).mean()


def _ewma_shifted(s: pd.Series, halflife: float) -> pd.Series:
    return s.shift(1).ewm(halflife=halflife, min_periods=1, adjust=False).mean()


def build_pre_match_features(
    matches: pd.DataFrame,
    *,
    rolling_windows: Iterable[int] = (3, 5, 8),
    ewma_halflives: Iterable[float] = (3, 5, 10),
    include_market: bool = True,
) -> tuple[pd.DataFrame, FeatureRegistry]:
    """
    Features pré-jogo por partida.
    Estatísticas HT/2T da própria partida NÃO entram.
    Odds FT pré-jogo podem entrar se include_market=True.
    """
    registry = FeatureRegistry()
    df = matches.sort_values("date", kind="mergesort").reset_index(drop=True).copy()
    long = _team_long(df)

    metrics = ["gf", "ga", "pts", "gd", "xg_for", "xg_against", "shots_for", "sot_for"]
    feat_parts: list[pd.DataFrame] = []

    for team, g in long.groupby("team", sort=False):
        g = g.copy()
        cols: dict[str, pd.Series] = {
            "date": g["date"],
            "team": g["team"],
            "opp": g["opp"],
            "is_home": g["is_home"],
        }
        for m in metrics:
            if m not in g.columns or g[m].isna().all():
                continue
            for w in rolling_windows:
                name = f"{m}_roll{w}"
                cols[name] = _rolling_shifted(g[m], int(w))
                registry.register(
                    FeatureMeta(name, "history", "result" if m in ("pts", "gf", "ga", "gd") else "xg",
                                window=int(w), transformation="rolling_mean")
                )
            for hl in ewma_halflives:
                name = f"{m}_ewm{int(hl)}"
                cols[name] = _ewma_shifted(g[m], float(hl))
                registry.register(
                    FeatureMeta(name, "history", "result", window=int(hl),
                                transformation="ewma")
                )
            # home/away splits
            for side, mask_val in (("home", 1), ("away", 0)):
                s = g[m].where(g["is_home"] == mask_val)
                name = f"{m}_{side}_roll5"
                cols[name] = _rolling_shifted(s, 5)
                registry.register(
                    FeatureMeta(name, "history", "result", window=5,
                                transformation="rolling_mean", home_away_specific=True)
                )
            # eficiência / regressão à média (com shrinkage implícito via rolling)
            if m == "gf" and "xg_for" in g.columns and g["xg_for"].notna().any():
                resid = g["gf"] - g["xg_for"]
                cols["gf_minus_xg_roll5"] = _rolling_shifted(resid, 5)
                registry.register(
                    FeatureMeta("gf_minus_xg_roll5", "history", "xg",
                                window=5, transformation="residual_rolling")
                )
        cols["days_since_last"] = g["date"].diff().dt.days
        registry.register(
            FeatureMeta("days_since_last", "calendar", "temporal",
                        transformation="diff_days")
        )
        feat_parts.append(pd.DataFrame(cols))

    team_feats = pd.concat(feat_parts, ignore_index=True)
    # Merge home/away features na linha da partida
    home_f = team_feats[team_feats["is_home"] == 1].drop(columns=["is_home", "opp"])
    away_f = team_feats[team_feats["is_home"] == 0].drop(columns=["is_home", "opp"])
    home_f = home_f.rename(columns={c: f"home_{c}" for c in home_f.columns if c not in ("date", "team")})
    away_f = away_f.rename(columns={c: f"away_{c}" for c in away_f.columns if c not in ("date", "team")})
    home_f = home_f.rename(columns={"team": "home_team"})
    away_f = away_f.rename(columns={"team": "away_team"})

    out = df.merge(home_f, on=["date", "home_team"], how="left")
    out = out.merge(away_f, on=["date", "away_team"], how="left")

    # mercado pré-jogo (de-vig básico)
    if include_market and {"odd_home_ft", "odd_draw_ft", "odd_away_ft"}.issubset(out.columns):
        from prob_ml.models.market import implied_1x2_devig

        p_h, p_d, p_a = implied_1x2_devig(
            out["odd_home_ft"], out["odd_draw_ft"], out["odd_away_ft"]
        )
        out["mkt_p_home"] = p_h
        out["mkt_p_draw"] = p_d
        out["mkt_p_away"] = p_a
        for n in ("mkt_p_home", "mkt_p_draw", "mkt_p_away"):
            registry.register(
                FeatureMeta(n, "market", "market", available_at="pre_kickoff")
            )

    # League mean goals (expanding, shifted by date)
    played = out["home_goals"].notna() & out["away_goals"].notna()
    tot = (out["home_goals"] + out["away_goals"]).where(played)
    out["league_avg_goals_pre"] = tot.shift(1).expanding(min_periods=1).mean()
    registry.register(
        FeatureMeta("league_avg_goals_pre", "history", "goals", transformation="expanding_mean")
    )

    try:
        from prob_ml.context_calendar import attach_context_to_matches

        out = attach_context_to_matches(out)
        for n in (
            "home_rest_days",
            "away_rest_days",
            "home_imp_classificatorias",
            "away_imp_classificatorias",
            "home_imp_oitavas",
            "away_imp_oitavas",
            "home_imp_quartas",
            "away_imp_quartas",
            "home_imp_semi",
            "away_imp_semi",
            "home_imp_final",
            "away_imp_final",
        ):
            if n in out.columns:
                registry.register(
                    FeatureMeta(n, "calendar", "context", available_at="pre_kickoff")
                )
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning("Contexto descanso/copas: %s", e)

    return out, registry


def assert_no_future_leakage(
    matches: pd.DataFrame,
    feature_builder,
    *,
    mutate_from_idx: int,
) -> None:
    """
    Teste de leakage: alterar partidas futuras não pode mudar features passadas.
    """
    base, _ = feature_builder(matches)
    alt = matches.copy()
    # Mutar gols futuros
    fut = alt.index >= mutate_from_idx
    alt.loc[fut, "home_goals"] = alt.loc[fut, "home_goals"].fillna(0) + 5
    alt.loc[fut, "away_goals"] = alt.loc[fut, "away_goals"].fillna(0) + 5
    if "home_xg" in alt.columns:
        alt.loc[fut, "home_xg"] = alt.loc[fut, "home_xg"].fillna(0) + 5
    changed, _ = feature_builder(alt)

    feat_cols = [
        c
        for c in base.columns
        if c.startswith(("home_", "away_", "league_", "mkt_"))
        and c
        not in {
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "home_xg",
            "away_xg",
            "home_shots",
            "away_shots",
            "home_sot",
            "away_sot",
            "home_corners",
            "away_corners",
            "home_possession",
            "away_possession",
            "home_big_chances",
            "away_big_chances",
        }
    ]
    past = base.index < mutate_from_idx
    numeric_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(base[c])]
    a = base.loc[past, numeric_cols].fillna(-999).to_numpy(dtype=float)
    b = changed.loc[past, numeric_cols].fillna(-999).to_numpy(dtype=float)
    if not np.allclose(a, b, equal_nan=True):
        diff = np.abs(a - b)
        raise AssertionError(
            f"Leakage detectado: max|Δ|={np.nanmax(diff):.6g} em features passadas"
        )
