"""Janela histórica (3 anos) e pesos exponenciais por recência (dias desde a partida)."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_HISTORY_YEARS = 3
DEFAULT_HALF_LIFE_DAYS = 120.0
DEFAULT_ELASTIC_NET_ALPHA = 0.01
DEFAULT_ELASTIC_NET_L1_RATIO = 0.5


def load_recency_settings(cfg: dict[str, Any] | None = None) -> dict[str, float | int]:
    if cfg is None:
        try:
            from prob_ml.config import load_config

            cfg = load_config()
        except Exception:
            cfg = {}
    data = (cfg or {}).get("data", {})
    return {
        "history_years": int(data.get("max_history_years", DEFAULT_HISTORY_YEARS)),
        "half_life_days": float(data.get("decay_half_life_days", DEFAULT_HALF_LIFE_DAYS)),
    }


def load_regression_settings(cfg: dict[str, Any] | None = None) -> dict[str, float | int]:
    base = load_recency_settings(cfg)
    if cfg is None:
        try:
            from prob_ml.config import load_config

            cfg = load_config()
        except Exception:
            cfg = {}
    reg = (cfg or {}).get("regression", {})
    return {
        **base,
        "elastic_net_alpha": float(reg.get("elastic_net_alpha", DEFAULT_ELASTIC_NET_ALPHA)),
        "elastic_net_l1_ratio": float(
            reg.get("elastic_net_l1_ratio", DEFAULT_ELASTIC_NET_L1_RATIO)
        ),
    }


def reference_date(ref: date | None = None) -> date:
    return ref or date.today()


def allowed_seasons(*, ref: date | None = None, years: int = DEFAULT_HISTORY_YEARS) -> set[int]:
    ref = reference_date(ref)
    y = ref.year
    return {y - i for i in range(max(1, years))}


def cutoff_date(*, ref: date | None = None, years: int = DEFAULT_HISTORY_YEARS) -> date:
    ref = reference_date(ref)
    return date(ref.year - (years - 1), 1, 1)


def parse_match_date(value: str | date | datetime | pd.Timestamp | None) -> date | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.date()
    s = str(value).strip()
    if not s or s.lower() in {"nan", "nat", "none", "-", ""}:
        return None
    try:
        return pd.Timestamp(s).date()
    except (TypeError, ValueError):
        return None


def days_since_match(match_date: date | None, ref: date | None = None) -> float:
    if match_date is None:
        return 0.0
    ref = reference_date(ref)
    return float(max((ref - match_date).days, 0))


def exponential_decay_weight(
    days_ago: float,
    *,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> float:
    if half_life_days <= 0:
        return 1.0
    if days_ago <= 0:
        return 1.0
    return float(0.5 ** (days_ago / half_life_days))


def weight_for_match_date(
    match_date: date | None,
    *,
    ref: date | None = None,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> float:
    return exponential_decay_weight(
        days_since_match(match_date, ref),
        half_life_days=half_life_days,
    )


def weights_for_match_dates(
    dates: list[str | date | datetime | pd.Timestamp | None],
    *,
    ref: date | None = None,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> np.ndarray:
    ref = reference_date(ref)
    return np.array(
        [
            weight_for_match_date(parse_match_date(d), ref=ref, half_life_days=half_life_days)
            for d in dates
        ],
        dtype=float,
    )


def filter_jogos_by_recency(
    jogos: list,
    *,
    years: int = DEFAULT_HISTORY_YEARS,
    ref: date | None = None,
) -> list:
    """Mantém jogos dos últimos ``years`` anos civis (inclui ano atual)."""
    cutoff = cutoff_date(ref=ref, years=years)
    out = []
    for j in jogos:
        d = parse_match_date(getattr(j, "data", "") or "")
        if d is None or d >= cutoff:
            out.append(j)
    return out


def filter_matches_dataframe(
    df: pd.DataFrame,
    *,
    years: int = DEFAULT_HISTORY_YEARS,
    ref: date | None = None,
) -> pd.DataFrame:
    """Filtra partidas para os últimos ``years`` anos (data e/ou season)."""
    if df.empty:
        return df
    ref = reference_date(ref)
    cutoff = cutoff_date(ref=ref, years=years)
    seasons = allowed_seasons(ref=ref, years=years)
    out = df.copy()
    mask = pd.Series(True, index=out.index)
    if "date" in out.columns:
        dt = pd.to_datetime(out["date"], errors="coerce")
        date_ok = dt.isna() | (dt.dt.normalize() >= pd.Timestamp(cutoff))
        mask &= date_ok
    if "season" in out.columns:
        s = pd.to_numeric(out["season"], errors="coerce")
        mask &= s.isna() | s.isin(seasons)
    return out.loc[mask].copy()


def attach_sample_weights(
    matches: pd.DataFrame,
    *,
    ref: date | None = None,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> pd.DataFrame:
    """Adiciona coluna ``sample_weight`` com decaimento exponencial por data."""
    out = matches.copy()
    if "date" not in out.columns:
        out["sample_weight"] = 1.0
        return out
    ref = reference_date(ref)
    weights = []
    for d in out["date"]:
        weights.append(
            weight_for_match_date(
                parse_match_date(d),
                ref=ref,
                half_life_days=half_life_days,
            )
        )
    out["sample_weight"] = np.asarray(weights, dtype=float)
    return out


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values, dtype=float)
    s = float(w.sum())
    if s <= 0:
        return float(v.mean()) if len(v) else 0.0
    return float(np.dot(v, w) / s)


def wls_lstsq(X: np.ndarray, y: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    """Mínimos quadrados ponderados via transformação sqrt(w)."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if weights is None:
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        return coef
    w = np.asarray(weights, dtype=float)
    w = np.clip(w, 1e-12, None)
    sw = np.sqrt(w)
    Xw = X * sw[:, np.newaxis]
    yw = y * sw
    coef, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
    return coef


def elastic_net_lstsq(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    alpha: float = DEFAULT_ELASTIC_NET_ALPHA,
    l1_ratio: float = DEFAULT_ELASTIC_NET_L1_RATIO,
) -> np.ndarray:
    """Elastic Net; ``X`` inclui coluna de intercepto (uns) quando aplicável."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.size == 0:
        return np.array([])
    n, p = X.shape
    if n == 0:
        return np.zeros(p)

    if weights is not None:
        sw = np.sqrt(np.clip(np.asarray(weights, dtype=float), 1e-12, None))
        X = X * sw[:, np.newaxis]
        y = y * sw

    has_intercept = p > 0 and np.allclose(X[:, 0], 1.0)
    X_fit = X[:, 1:] if has_intercept else X

    if X_fit.shape[1] == 0:
        intercept = float(np.mean(y)) if has_intercept else 0.0
        return np.array([intercept]) if has_intercept else np.array([])

    try:
        from sklearn.linear_model import ElasticNet
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return wls_lstsq(X, y, None)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_fit)
    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=8000,
        fit_intercept=True,
    )
    model.fit(Xs, y)
    scale = scaler.scale_
    mean = scaler.mean_
    coef_feat = model.coef_ / scale
    intercept = float(model.intercept_ - np.dot(model.coef_, mean / scale))
    if has_intercept:
        return np.concatenate([[intercept], coef_feat])
    return coef_feat
