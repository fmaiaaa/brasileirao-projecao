"""Ratings temporais (Elo e variantes simples)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EloState:
    ratings: dict[str, float]
    history: list[dict]


def expected_score(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def update_elo(
    matches: pd.DataFrame,
    *,
    k: float = 20.0,
    home_adv: float = 60.0,
    initial: float = 1500.0,
    season_carry: float = 0.7,
    league_mean: float = 1500.0,
) -> tuple[pd.DataFrame, EloState]:
    """
    Elo resultado com home advantage.
    Ratings pré-jogo são gravados ANTES do update (leakage-safe).
    """
    df = matches.sort_values("date", kind="mergesort").reset_index(drop=True).copy()
    ratings: dict[str, float] = {}
    hist: list[dict] = []
    pre_h: list[float] = []
    pre_a: list[float] = []
    last_season = None

    for _, row in df.iterrows():
        season = row.get("season")
        if last_season is not None and season is not None and season != last_season:
            for t in list(ratings.keys()):
                ratings[t] = season_carry * ratings[t] + (1.0 - season_carry) * league_mean
        last_season = season

        h, a = str(row["home_team"]), str(row["away_team"])
        ratings.setdefault(h, initial)
        ratings.setdefault(a, initial)
        rh, ra = ratings[h], ratings[a]
        pre_h.append(rh)
        pre_a.append(ra)

        hg, ag = row.get("home_goals"), row.get("away_goals")
        if pd.isna(hg) or pd.isna(ag):
            continue
        hg, ag = float(hg), float(ag)
        if hg > ag:
            sh, sa = 1.0, 0.0
        elif hg < ag:
            sh, sa = 0.0, 1.0
        else:
            sh, sa = 0.5, 0.5
        eh = expected_score(rh + home_adv, ra)
        ea = 1.0 - eh
        ratings[h] = rh + k * (sh - eh)
        ratings[a] = ra + k * (sa - ea)
        hist.append({"date": row["date"], "home": h, "away": a, "rh": ratings[h], "ra": ratings[a]})

    out = df.copy()
    out["elo_home_pre"] = pre_h
    out["elo_away_pre"] = pre_a
    out["elo_home_win_prob"] = [
        expected_score(h + home_adv, a) for h, a in zip(pre_h, pre_a)
    ]
    return out, EloState(ratings=ratings, history=hist)


def elo_xg(
    matches: pd.DataFrame,
    *,
    k: float = 15.0,
    home_adv: float = 40.0,
) -> pd.DataFrame:
    """Elo usando xG como proxy de margem (quando disponível)."""
    df = matches.copy()
    if "home_xg" not in df.columns or df["home_xg"].isna().all():
        out, _ = update_elo(df, k=k, home_adv=home_adv)
        return out.rename(
            columns={
                "elo_home_pre": "elo_xg_home_pre",
                "elo_away_pre": "elo_xg_away_pre",
                "elo_home_win_prob": "elo_xg_home_win_prob",
            }
        )
    # Mapear xG para score 0/0.5/1 via comparação
    tmp = df.copy()
    tmp["home_goals"] = np.where(
        tmp["home_xg"] > tmp["away_xg"] + 0.05,
        1,
        np.where(tmp["home_xg"] < tmp["away_xg"] - 0.05, 0, 1),
    )
    # fake away goals complementary for update_elo win logic
    tmp["away_goals"] = 1 - tmp["home_goals"]
    tmp.loc[np.isclose(df["home_xg"], df["away_xg"], atol=0.05), "home_goals"] = 1
    tmp.loc[np.isclose(df["home_xg"], df["away_xg"], atol=0.05), "away_goals"] = 1
    out, _ = update_elo(tmp, k=k, home_adv=home_adv)
    df = df.copy()
    df["elo_xg_home_pre"] = out["elo_home_pre"]
    df["elo_xg_away_pre"] = out["elo_away_pre"]
    return df
