"""Seleção de features (dentro do fold temporal)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def variance_filter(X: pd.DataFrame, threshold: float = 1e-8) -> list[str]:
    cols = []
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]) and float(X[c].var(skipna=True) or 0) > threshold:
            cols.append(c)
    return cols


def correlation_filter(X: pd.DataFrame, cols: list[str], threshold: float = 0.95) -> list[str]:
    if len(cols) < 2:
        return cols
    corr = X[cols].fillna(0).corr().abs()
    keep: list[str] = []
    dropped: set[str] = set()
    for c in cols:
        if c in dropped:
            continue
        keep.append(c)
        for o in cols:
            if o != c and o not in dropped and corr.loc[c, o] >= threshold:
                dropped.add(o)
    return keep


def select_features(
    features: pd.DataFrame,
    y_home: pd.Series,
    y_away: pd.Series,
    *,
    max_features: int = 30,
) -> tuple[list[str], dict[str, float]]:
    """Filtros + correlação com target (dentro do train fold)."""
    cand = [
        c
        for c in features.columns
        if pd.api.types.is_numeric_dtype(features[c])
        and c
        not in {
            "home_goals",
            "away_goals",
            "round",
            "season",
            "match_id",
        }
        and not c.endswith("_team")
    ]
    cand = variance_filter(features[cand]) if cand else []
    cand = correlation_filter(features, cand)
    scores: dict[str, float] = {}
    y = (y_home.fillna(0) + y_away.fillna(0)).astype(float)
    for c in cand:
        s = features[c].astype(float)
        mask = s.notna() & y.notna()
        if mask.sum() < 10:
            scores[c] = 0.0
            continue
        scores[c] = float(abs(np.corrcoef(s[mask], y[mask])[0, 1]))
        if np.isnan(scores[c]):
            scores[c] = 0.0
    ranked = sorted(scores, key=lambda k: -scores[k])[:max_features]
    return ranked, scores
