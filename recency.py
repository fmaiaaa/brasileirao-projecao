"""Janela das últimas 38 rodadas e pesos por distância de rodada (decaimento progressivo)."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd

DEFAULT_HISTORY_ROUNDS = 38
DEFAULT_HALF_LIFE_ROUNDS = 12.0  # legado (prob_ml antigo)
DEFAULT_ELASTIC_NET_ALPHA = 0.01
DEFAULT_ELASTIC_NET_L1_RATIO = 0.5
DEFAULT_TRAINING_YEARS = 3
# Pesos de importância (estimativas): últimas 5 rodadas 100%→90%;
# 6ª até 1ª rodada da temporada 90%→25%; anos passados 25%→0%.
W_RECENT_ROUNDS = 5
W_MAX = 1.0
W_AFTER_RECENT = 0.90
W_SEASON_FLOOR = 0.25
W_PAST_MAX = 0.25
W_PAST_MIN = 0.0
SERIE_A = "serie_a"
SERIE_B = "serie_b"

JanelaTreino = Literal["2026", "ultimas_38_rodadas", "ultimos_3_anos", "base_completa"]

JANELA_TREINO_LABELS: dict[JanelaTreino, str] = {
    "2026": "Só 2026",
    "ultimas_38_rodadas": "Últimas 38 rodadas",
    "ultimos_3_anos": "Últimos 3 anos",
    "base_completa": "Base completa",
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
    """Legado — preferir ``importance_weight_by_rounds_ago``."""
    if half_life_rounds <= 0:
        return 1.0
    if rounds_ago <= 0:
        return 1.0
    return float(0.5 ** (rounds_ago / half_life_rounds))


def importance_weight_by_rounds_ago(
    rounds_ago: float,
    *,
    rounds_ago_season_start: float | None = None,
    is_past_season: bool = False,
    past_year_fraction: float = 0.0,
) -> float:
    """
    Importância estimada do jogo no treino (0–1).

    Temporada atual:
      - rodadas 0–4 (últimas 5): 100% → 90%
      - da 6ª até a 1ª rodada: 90% → 25%

    Anos passados (conforme janela):
      - 25% → 0% (mais recente → mais antigo)
    """
    if is_past_season:
        frac = min(max(float(past_year_fraction), 0.0), 1.0)
        return max(W_PAST_MIN, W_PAST_MAX * (1.0 - frac))

    ra = max(float(rounds_ago), 0.0)
    if ra <= W_RECENT_ROUNDS - 1:
        return W_MAX - 0.025 * ra  # 1.00, 0.975, 0.95, 0.925, 0.90

    ra_first = float(rounds_ago_season_start) if rounds_ago_season_start is not None else ra
    ra_first = max(ra_first, float(W_RECENT_ROUNDS))
    span = max(ra_first - float(W_RECENT_ROUNDS), 1.0)
    t = min(max(ra - float(W_RECENT_ROUNDS), 0.0) / span, 1.0)
    return W_AFTER_RECENT - t * (W_AFTER_RECENT - W_SEASON_FLOOR)


def _oldest_season_for_janela(
    janela: JanelaTreino | None,
    *,
    current_season: int,
    ano_calendario: int | None = None,
) -> int:
    cal = int(ano_calendario or current_season)
    if janela == "2026":
        return int(ano_calendario or current_season)
    if janela == "2025":  # legado
        return 2025
    if janela == "ultimos_3_anos":
        return cal - DEFAULT_TRAINING_YEARS
    return current_season - 1


def importance_weight_observation(
    season: int,
    round_num: int,
    *,
    current_season: int,
    r_latest: int,
    janela: JanelaTreino | None = None,
    ano_calendario: int | None = None,
) -> float:
    """Peso para uma observação (season, round) vs. calendário-alvo."""
    season = int(season)
    round_num = int(round_num)
    current_season = int(current_season)
    r_latest = int(r_latest)

    if season > current_season:
        return 0.0

    if season < current_season:
        oldest = _oldest_season_for_janela(
            janela, current_season=current_season, ano_calendario=ano_calendario
        )
        if season < oldest:
            return 0.0
        span = max(current_season - oldest, 1)
        past_frac = (current_season - season - 1) / span
        past_frac = min(max(past_frac, 0.0), 1.0)
        ra_in_season = max(38 - round_num, 0)
        intra = 0.15 * (ra_in_season / 37.0)
        return max(
            W_PAST_MIN,
            W_PAST_MAX * (1.0 - past_frac) - intra * W_PAST_MAX,
        )

    rounds_ago = max(r_latest - round_num, 0)
    rounds_ago_first = max(r_latest - 1, W_RECENT_ROUNDS)
    return importance_weight_by_rounds_ago(
        rounds_ago,
        rounds_ago_season_start=rounds_ago_first,
        is_past_season=False,
    )


def weights_for_panel_observations(
    seasons: list[int],
    rodadas: list[float],
    *,
    current_season: int,
    r_latest: int | None = None,
    janela: JanelaTreino | None = None,
    ano_calendario: int | None = None,
) -> np.ndarray:
    if r_latest is None:
        r_latest = int(max(rodadas)) if rodadas else 38
    return np.array(
        [
            importance_weight_observation(
                int(s),
                int(r),
                current_season=int(current_season),
                r_latest=int(r_latest),
                janela=janela,
                ano_calendario=ano_calendario,
            )
            for s, r in zip(seasons, rodadas)
        ],
        dtype=float,
    )


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


def filter_matches_by_janela(
    df: pd.DataFrame,
    janela: JanelaTreino,
    *,
    ano_calendario: int,
    calendar_teams: Iterable[str] | None = None,
    history_rounds: int = DEFAULT_HISTORY_ROUNDS,
) -> pd.DataFrame:
    """Filtra partidas FPT conforme janela de treino (espelha ``preparar_blocos_treino_janela``)."""
    if janela == "ultimas_38_rodadas":
        return filter_matches_dataframe(
            df,
            history_rounds=history_rounds,
            calendar_teams=calendar_teams,
            current_season=int(ano_calendario),
        )
    if df.empty:
        return df
    prep = _prepare_competition(df)
    played = _played_mask(prep)
    prep = prep.loc[played].copy()
    if prep.empty:
        return prep.drop(columns=[c for c in prep.columns if c.startswith("_")], errors="ignore")

    if janela == "2026":
        sa_mask = (prep["_comp"] == SERIE_A) & (prep["season"] == int(ano_calendario))
        sb_season = int(ano_calendario) - 1
    elif janela == "2025":
        sa_mask = (prep["_comp"] == SERIE_A) & (prep["season"] == 2025)
        sb_season = 2024
    elif janela == "base_completa":
        sa_mask = prep["_comp"] == SERIE_A
        sb_season = int(ano_calendario) - 1
    elif janela == "ultimos_3_anos":
        anos = anos_janela_tres_anos(int(ano_calendario), n_anos=3)
        sa_mask = (prep["_comp"] == SERIE_A) & prep["season"].isin(anos)
        sb_season = min(anos) - 1
    else:
        return filter_matches_dataframe(
            df,
            history_rounds=history_rounds,
            calendar_teams=calendar_teams,
            current_season=int(ano_calendario),
        )

    promoted = detect_promoted_teams(
        prep, calendar_teams=calendar_teams, current_season=int(ano_calendario)
    )
    if promoted:
        sb_mask = (prep["_comp"] == SERIE_B) & (prep["season"] == sb_season) & (
            prep["home_team"].astype(str).isin(promoted)
            | prep["away_team"].astype(str).isin(promoted)
        )
    else:
        sb_mask = pd.Series(False, index=prep.index)

    out = prep.loc[sa_mask | sb_mask].copy()
    return out.drop(columns=[c for c in out.columns if c.startswith("_")], errors="ignore")


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
    janela: JanelaTreino | None = None,
    ano_calendario: int | None = None,
) -> pd.DataFrame:
    """Coluna ``sample_weight`` com decaimento por distância de rodada."""
    out = _prepare_competition(matches)
    _, idx_map, latest_idx = _round_timeline(out, competition=SERIE_A)
    if not idx_map:
        out["sample_weight"] = 1.0
        return out.drop(columns=[c for c in out.columns if c.startswith("_")], errors="ignore")

    if current_season is None:
        current_season = max(k[0] for k in idx_map)

    latest_key = max(idx_map, key=lambda k: idx_map[k])
    current_season, r_latest = int(latest_key[0]), int(latest_key[1])

    weights = []
    for _, row in out.iterrows():
        comp = str(row["_comp"])
        season = int(row["season"])
        rnd = int(row["round"])
        if comp == SERIE_A:
            w = importance_weight_observation(
                season,
                rnd,
                current_season=int(current_season),
                r_latest=int(r_latest),
                janela=janela,
                ano_calendario=int(ano_calendario or current_season),
            )
        else:
            ra = _rounds_ago_from_timeline(
                season, rnd, idx_map, latest_idx, competition=comp
            )
            w = importance_weight_by_rounds_ago(
                ra,
                rounds_ago_season_start=float(DEFAULT_HISTORY_ROUNDS),
                is_past_season=True,
                past_year_fraction=0.5,
            )
        weights.append(w)
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
    current_season: int | None = None,
    janela: JanelaTreino | None = None,
    ano_calendario: int | None = None,
) -> float:
    del half_life_rounds  # legado
    if not getattr(jogo, "jogado", False):
        return 1.0
    if r_latest is None:
        r_latest = int(getattr(jogo, "r", 0))
    season = current_season
    if season is None:
        d = parse_match_date(getattr(jogo, "data", "") or "")
        season = d.year if d else reference_date().year
    return importance_weight_observation(
        int(season),
        int(getattr(jogo, "r", 0)),
        current_season=int(season),
        r_latest=int(r_latest),
        janela=janela,
        ano_calendario=ano_calendario or season,
    )


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
    seasons: list[int] | None = None,
    current_season: int | None = None,
    janela: JanelaTreino | None = None,
    ano_calendario: int | None = None,
) -> np.ndarray:
    del half_life_rounds
    r_list = list(rodadas)
    if seasons is not None and len(seasons) == len(r_list):
        cs = int(current_season or (max(seasons) if seasons else reference_date().year))
        rl = int(r_latest) if r_latest is not None else int(max(r_list) if r_list else 38)
        return weights_for_panel_observations(
            [int(s) for s in seasons],
            r_list,
            current_season=cs,
            r_latest=rl,
            janela=janela,
            ano_calendario=ano_calendario,
        )
    r = np.asarray(rodadas, dtype=float)
    if r_latest is None:
        r_latest = float(np.nanmax(r)) if len(r) else 0.0
    ra_first = max(float(r_latest) - 1.0, float(W_RECENT_ROUNDS))
    return np.array(
        [
            importance_weight_by_rounds_ago(
                max(float(r_latest) - float(rv), 0.0),
                rounds_ago_season_start=ra_first,
            )
            for rv in r
        ],
        dtype=float,
    )


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
