"""Janela das últimas 38 rodadas e pesos por distância de rodada (decaimento progressivo)."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd

DEFAULT_HISTORY_ROUNDS = 38
DEFAULT_HALF_LIFE_ROUNDS = 12.0
DEFAULT_ELASTIC_NET_ALPHA = 0.01
DEFAULT_ELASTIC_NET_L1_RATIO = 0.5
DEFAULT_TRAINING_YEARS = 3
SERIE_A = "serie_a"
SERIE_B = "serie_b"

JanelaTreino = Literal["2025", "ultimas_38_rodadas", "ultimos_3_anos"]

JANELA_TREINO_LABELS: dict[JanelaTreino, str] = {
    "2025": "Só 2025",
    "ultimas_38_rodadas": "Últimas 38 rodadas",
    "ultimos_3_anos": "Últimos 3 anos",
}


def anos_janela_tres_anos(ano_calendario: int, *, n_anos: int = DEFAULT_TRAINING_YEARS) -> list[int]:
    """Temporadas completas anteriores ao calendário-alvo (ex.: 2026 → 2023–2025)."""
    n = max(1, int(n_anos))
    return list(range(int(ano_calendario) - n, int(ano_calendario)))


def load_recency_settings(cfg: dict[str, Any] | None = None) -> dict[str, float | int]:
    if cfg is None:
        try:
            from prob_ml.config import load_config

            cfg = load_config()
        except Exception:
            cfg = {}
    data = (cfg or {}).get("data", {})
    return {
        "history_rounds": int(data.get("max_history_rounds", DEFAULT_HISTORY_ROUNDS)),
        "half_life_rounds": float(
            data.get("decay_half_life_rounds", DEFAULT_HALF_LIFE_ROUNDS)
        ),
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


def allowed_seasons(*, ref: date | None = None, years: int = 2) -> set[int]:
    """Temporadas FPT necessárias: ano atual + anterior (38 rodadas + Série B promovidos)."""
    ref = reference_date(ref)
    y = ref.year
    return {y, y - 1}


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
        return pd.Timestamp(s, dayfirst=True).date()
    except (TypeError, ValueError):
        try:
            return pd.Timestamp(s).date()
        except (TypeError, ValueError):
            return None


def normalize_competition(raw: str) -> str:
    s = str(raw).strip().lower()
    if "women" in s or "femin" in s:
        return "brasileiro_women"
    if "serie b" in s or s == "serie_b":
        return SERIE_B
    if "serie a" in s or "betano" in s or s == "serie_a":
        return SERIE_A
    return s.replace(" ", "_").replace("-", "_")


def exponential_decay_rounds(
    rounds_ago: float,
    *,
    half_life_rounds: float = DEFAULT_HALF_LIFE_ROUNDS,
) -> float:
    if half_life_rounds <= 0:
        return 1.0
    if rounds_ago <= 0:
        return 1.0
    return float(0.5 ** (rounds_ago / half_life_rounds))


def _played_mask(df: pd.DataFrame) -> pd.Series:
    if "home_goals" not in df.columns or "away_goals" not in df.columns:
        return pd.Series(True, index=df.index)
    return df["home_goals"].notna() & df["away_goals"].notna()


def _prepare_competition(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "competition" not in out.columns:
        out["competition"] = SERIE_A
    out["_comp"] = out["competition"].map(normalize_competition)
    if "season" not in out.columns:
        out["season"] = reference_date().year
    out["season"] = pd.to_numeric(out["season"], errors="coerce")
    if "round" not in out.columns:
        out["round"] = 1
    out["round"] = pd.to_numeric(out["round"], errors="coerce").fillna(1).astype(int)
    if "date" in out.columns:
        out["_dt"] = pd.to_datetime(out["date"], errors="coerce", dayfirst=True)
    else:
        out["_dt"] = pd.NaT
    return out


def _round_timeline(
    df: pd.DataFrame,
    *,
    competition: str = SERIE_A,
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], int], int]:
    """Ordem cronológica de (season, round) e índice da rodada mais recente."""
    sub = df[(df["_comp"] == competition) & _played_mask(df)].copy()
    if sub.empty:
        return [], {}, 0
    grp = (
        sub.groupby(["season", "round"], dropna=False)
        .agg(_dt=("_dt", "max"))
        .reset_index()
        .sort_values(["_dt", "season", "round"], kind="mergesort")
    )
    keys = [(int(row["season"]), int(row["round"])) for _, row in grp.iterrows()]
    idx_map = {k: i for i, k in enumerate(keys)}
    latest = max(idx_map.values()) if idx_map else 0
    return keys, idx_map, latest


def _teams_in_season(df: pd.DataFrame, *, competition: str, season: int) -> set[str]:
    sub = df[(df["_comp"] == competition) & (df["season"] == season) & _played_mask(df)]
    if sub.empty:
        return set()
    home = sub["home_team"].astype(str).str.strip()
    away = sub["away_team"].astype(str).str.strip()
    return set(home) | set(away)


def detect_promoted_teams(
    df: pd.DataFrame,
    *,
    calendar_teams: Iterable[str] | None = None,
    current_season: int | None = None,
) -> set[str]:
    """
    Times na Série A atual que não estavam na Série A na temporada anterior
    (recém-promovidos — histórico na Série B).
    """
    prep = _prepare_competition(df)
    if current_season is None:
        sa = prep[prep["_comp"] == SERIE_A]
        if sa["season"].notna().any():
            current_season = int(sa["season"].max())
        else:
            current_season = reference_date().year
    prev_season = int(current_season) - 1
    prev_a = _teams_in_season(prep, competition=SERIE_A, season=prev_season)
    cur_a = _teams_in_season(prep, competition=SERIE_A, season=current_season)
    if calendar_teams:
        cur_set = {str(t).strip() for t in calendar_teams}
        promoted = cur_set - prev_a
    else:
        promoted = cur_a - prev_a
    return {t for t in promoted if t and t.lower() != "nan"}


def _rounds_ago_from_timeline(
    season: int,
    round_num: int,
    idx_map: dict[tuple[int, int], int],
    latest_idx: int,
    *,
    competition: str = SERIE_A,
) -> float:
    key = (int(season), int(round_num))
    if key in idx_map:
        return float(max(latest_idx - idx_map[key], 0))
    if competition == SERIE_B:
        # Temporada anterior na Série B: mais distante que a janela de 38 rodadas
        return float(latest_idx + max(1, DEFAULT_HISTORY_ROUNDS - int(round_num) + 1))
    return float(latest_idx + 1)


def filter_matches_dataframe(
    df: pd.DataFrame,
    *,
    history_rounds: int = DEFAULT_HISTORY_ROUNDS,
    calendar_teams: Iterable[str] | None = None,
    current_season: int | None = None,
) -> pd.DataFrame:
    """
    Mantém:
      - Série A: últimas ``history_rounds`` rodadas (cronológicas entre temporadas)
      - Série B: jogos da temporada anterior envolvendo times promovidos
    """
    if df.empty:
        return df
    prep = _prepare_competition(df)
    timeline, idx_map, _latest = _round_timeline(prep, competition=SERIE_A)
    if not timeline:
        return prep.drop(columns=[c for c in prep.columns if c.startswith("_")], errors="ignore")
    keep_keys = set(timeline[-max(1, history_rounds) :])
    if current_season is None:
        current_season = int(timeline[-1][0])
    promoted = detect_promoted_teams(
        prep, calendar_teams=calendar_teams, current_season=current_season
    )
    prev_season = int(current_season) - 1

    sa_mask = (prep["_comp"] == SERIE_A) & prep.apply(
        lambda r: (int(r["season"]), int(r["round"])) in keep_keys, axis=1
    )
    if promoted:
        sb_mask = (prep["_comp"] == SERIE_B) & (prep["season"] == prev_season) & (
            prep["home_team"].astype(str).isin(promoted)
            | prep["away_team"].astype(str).isin(promoted)
        )
    else:
        sb_mask = pd.Series(False, index=prep.index)

    out = prep.loc[sa_mask | sb_mask].copy()
    return out.drop(columns=[c for c in out.columns if c.startswith("_")], errors="ignore")


def attach_sample_weights(
    matches: pd.DataFrame,
    *,
    half_life_rounds: float = DEFAULT_HALF_LIFE_ROUNDS,
    calendar_teams: Iterable[str] | None = None,
    current_season: int | None = None,
) -> pd.DataFrame:
    """Coluna ``sample_weight`` com decaimento por distância de rodada."""
    out = _prepare_competition(matches)
    _, idx_map, latest_idx = _round_timeline(out, competition=SERIE_A)
    if not idx_map:
        out["sample_weight"] = 1.0
        return out.drop(columns=[c for c in out.columns if c.startswith("_")], errors="ignore")

    if current_season is None:
        current_season = max(k[0] for k in idx_map)

    weights = []
    for _, row in out.iterrows():
        comp = str(row["_comp"])
        ra = _rounds_ago_from_timeline(
            int(row["season"]),
            int(row["round"]),
            idx_map,
            latest_idx,
            competition=comp,
        )
        weights.append(exponential_decay_rounds(ra, half_life_rounds=half_life_rounds))
    out["sample_weight"] = np.asarray(weights, dtype=float)
    return out.drop(columns=[c for c in out.columns if c.startswith("_")], errors="ignore")


def _latest_played_round(jogos: list) -> int:
    played = [getattr(j, "r", 0) for j in jogos if getattr(j, "jogado", False)]
    return max(played) if played else 0


def filter_jogos_by_round_window(
    jogos: list,
    *,
    n_rounds: int = DEFAULT_HISTORY_ROUNDS,
) -> list:
    """Mantém jogos das últimas ``n_rounds`` rodadas disputadas no bloco."""
    played_rounds = sorted({int(j.r) for j in jogos if getattr(j, "jogado", False)})
    if not played_rounds:
        return list(jogos)
    keep = set(played_rounds[-max(1, n_rounds) :])
    return [j for j in jogos if not getattr(j, "jogado", False) or int(j.r) in keep]


# Alias legado
filter_jogos_by_recency = filter_jogos_by_round_window


def weight_for_jogo(
    jogo,
    *,
    r_latest: int | None = None,
    half_life_rounds: float = DEFAULT_HALF_LIFE_ROUNDS,
) -> float:
    if not getattr(jogo, "jogado", False):
        return 1.0
    if r_latest is None:
        r_latest = int(getattr(jogo, "r", 0))
    rounds_ago = max(int(r_latest) - int(getattr(jogo, "r", 0)), 0)
    return exponential_decay_rounds(rounds_ago, half_life_rounds=half_life_rounds)


def weights_for_jogos(
    jogos: list,
    *,
    half_life_rounds: float = DEFAULT_HALF_LIFE_ROUNDS,
) -> np.ndarray:
    r_latest = _latest_played_round(jogos)
    return np.array(
        [weight_for_jogo(j, r_latest=r_latest, half_life_rounds=half_life_rounds) for j in jogos],
        dtype=float,
    )


def weights_for_rounds_ago(
    rounds_ago: Iterable[float],
    *,
    half_life_rounds: float = DEFAULT_HALF_LIFE_ROUNDS,
) -> np.ndarray:
    return np.array(
        [
            exponential_decay_rounds(float(r), half_life_rounds=half_life_rounds)
            for r in rounds_ago
        ],
        dtype=float,
    )


def weights_for_panel_rounds(
    rodadas: np.ndarray | list[float],
    *,
    r_latest: float | None = None,
    half_life_rounds: float = DEFAULT_HALF_LIFE_ROUNDS,
) -> np.ndarray:
    r = np.asarray(rodadas, dtype=float)
    if r_latest is None:
        r_latest = float(np.nanmax(r)) if len(r) else 0.0
    rounds_ago = np.maximum(r_latest - r, 0.0)
    return weights_for_rounds_ago(rounds_ago, half_life_rounds=half_life_rounds)


def weights_for_match_dates(
    dates: list[str | date | datetime | pd.Timestamp | None],
    *,
    half_life_rounds: float = DEFAULT_HALF_LIFE_ROUNDS,
    **_: Any,
) -> np.ndarray:
    """Compat: pesos uniformes decrescentes por ordem cronológica das datas."""
    parsed = [parse_match_date(d) for d in dates]
    order = sorted(range(len(parsed)), key=lambda i: parsed[i] or date.min)
    n = len(order)
    w = np.ones(n, dtype=float)
    for rank, i in enumerate(order):
        rounds_ago = float((n - 1) - rank)
        w[i] = exponential_decay_rounds(rounds_ago, half_life_rounds=half_life_rounds)
    return w


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values, dtype=float)
    s = float(w.sum())
    if s <= 0:
        return float(v.mean()) if len(v) else 0.0
    return float(np.dot(v, w) / s)


def wls_lstsq(X: np.ndarray, y: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
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
