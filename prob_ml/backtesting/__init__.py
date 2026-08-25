"""Splitters temporais (proibido random k-fold)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TemporalSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray
    name: str = ""


def expanding_season_splits(
    matches: pd.DataFrame,
    *,
    min_train_seasons: int = 1,
) -> list[TemporalSplit]:
    """Outer-style: treina temporadas anteriores, testa a seguinte."""
    if "season" not in matches.columns or matches["season"].isna().all():
        return date_expanding_splits(matches, n_splits=3)
    seasons = sorted(matches["season"].dropna().unique())
    splits = []
    for i in range(min_train_seasons, len(seasons)):
        train_seasons = seasons[:i]
        test_season = seasons[i]
        tr = matches.index[matches["season"].isin(train_seasons)].to_numpy()
        te = matches.index[matches["season"] == test_season].to_numpy()
        # só jogos com placar no teste
        te = te[matches.loc[te, "home_goals"].notna().to_numpy()]
        if len(tr) and len(te):
            splits.append(TemporalSplit(tr, te, name=f"test_{test_season}"))
    return splits


def date_expanding_splits(
    matches: pd.DataFrame, n_splits: int = 3
) -> list[TemporalSplit]:
    played = matches.dropna(subset=["home_goals", "away_goals"]).sort_values("date")
    idx = played.index.to_numpy()
    if len(idx) < n_splits + 5:
        return []
    folds = np.array_split(idx, n_splits + 1)
    splits = []
    train = folds[0]
    for i, val in enumerate(folds[1:], start=1):
        splits.append(TemporalSplit(train.copy(), val.copy(), name=f"fold_{i}"))
        train = np.concatenate([train, val])
    return splits


def nested_inner_splits(
    train_idx: np.ndarray, matches: pd.DataFrame, n_inner: int = 3
) -> list[TemporalSplit]:
    sub = matches.loc[train_idx].sort_values("date")
    return date_expanding_splits(sub, n_splits=n_inner)
