"""
Contexto pré-jogo: dias de descanso e jogos importantes à frente.

Categorias de importância (mais grave vence no horizonte):
  Não tem | Classificatórias | Oitavas | Quartas | Semi | Final

Calendário de copas: FPT Libertadores/Sul-Americana (+ parquet local se existir).
Olhar para o futuro só usa a DATA do jogo de copa (calendário conhecido), nunca o placar.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CUP_CSV = ROOT / "dados" / "fpt_cups.csv"
HORIZON_DAYS = 10
DEFAULT_REST = 7.0

IMPORTANCE_LEVELS = [
    "Não tem",
    "Classificatórias",
    "Oitavas",
    "Quartas",
    "Semi",
    "Final",
]
IMP_RANK = {k: i for i, k in enumerate(IMPORTANCE_LEVELS)}
IMP_DUMMY_COLS = [
    "imp_classificatorias",
    "imp_oitavas",
    "imp_quartas",
    "imp_semi",
    "imp_final",
]
IMP_DUMMY_LABELS = {
    "imp_classificatorias": "Classificatórias",
    "imp_oitavas": "Oitavas",
    "imp_quartas": "Quartas",
    "imp_semi": "Semi",
    "imp_final": "Final",
}

FPT_CUP_LEAGUES = [
    {"slug": "south-america/copa-libertadores", "competition": "libertadores"},
    {"slug": "south-america/copa-sudamericana", "competition": "sudamericana"},
]


def _strip(s: str) -> str:
    t = "".join(
        c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c)
    )
    return re.sub(r"\s+", " ", t.strip().lower())


def classify_stage(round_label: Any, competition: str = "") -> str:
    s = _strip(str(round_label or ""))
    comp = _strip(competition)
    if not s or s in {"nan", "none", ""}:
        return "Não tem"
    if any(x in s for x in ("final", "play off final")) and "semi" not in s:
        if "quarter" in s:
            return "Quartas"
        return "Final"
    if "semi" in s:
        return "Semi"
    if any(x in s for x in ("quarter", "quartas", "qf")):
        return "Quartas"
    if any(
        x in s
        for x in (
            "round of 16",
            "oitavas",
            "last 16",
            "1/8",
            "eighth",
        )
    ):
        return "Oitavas"
    if any(
        x in s
        for x in (
            "qualif",
            "group",
            "1st round",
            "2nd round",
            "3rd round",
            "prelim",
            "play in",
            "classific",
        )
    ):
        return "Classificatórias"
    if "play off" in s or "playoff" in s or s == "knockout":
        return "Oitavas"
    if "libertadores" in comp or "sudamericana" in comp:
        return "Classificatórias"
    return "Não tem"


def _refine_playoffs(df: pd.DataFrame) -> pd.Series:
    """Se Round só diz Play Offs, infere Final/Semi/Quartas/Oitavas pela ordem das datas."""
    out = df["importance"].copy()
    if "competition" not in df.columns or "season" not in df.columns:
        return out
    mask = df["round_raw"].astype(str).str.contains("play off", case=False, na=False)
    for (_, _), g in df.loc[mask].groupby(["competition", "season"], dropna=False):
        dates = sorted(pd.to_datetime(g["date"], errors="coerce").dropna().unique())
        if not dates:
            continue
        n = len(dates)
        mapping: dict[Any, str] = {}
        # último(s) dia(s) = Final; anteriores em blocos
        mapping[dates[-1]] = "Final"
        if n >= 2:
            mapping[dates[-2]] = "Semi"
        if n >= 3:
            mapping[dates[-3]] = "Semi"
        if n >= 4:
            mapping[dates[-4]] = "Quartas"
        if n >= 5:
            mapping[dates[-5]] = "Quartas"
        for d in dates[:-5] if n > 5 else []:
            mapping[d] = "Oitavas"
        for idx, row in g.iterrows():
            dt = pd.to_datetime(row["date"], errors="coerce")
            if dt in mapping:
                out.at[idx] = mapping[dt]
    return out


def _events_from_fpt_like(raw: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in raw.columns}
    def pick(*names: str) -> pd.Series:
        for n in names:
            if n.lower() in cols:
                return raw[cols[n.lower()]]
            if n in raw.columns:
                return raw[n]
        return pd.Series([np.nan] * len(raw))

    round_raw = pick("Round_raw", "round_raw", "Round", "round")
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(pick("Date", "date", "kickoff"), errors="coerce", dayfirst=True),
            "home_team": pick("Home", "home_team").astype(str),
            "away_team": pick("Away", "away_team").astype(str),
            "round_raw": round_raw.astype(str),
            "competition": pick("fpt_competition", "competition", "League", "league_name").astype(str),
            "season": pd.to_numeric(pick("Season", "season", "season_year"), errors="coerce"),
        }
    )
    df = df.dropna(subset=["date"])
    df["home_key"] = df["home_team"].map(_strip)
    df["away_key"] = df["away_team"].map(_strip)
    df["importance"] = [
        classify_stage(r, c) for r, c in zip(df["round_raw"], df["competition"])
    ]
    df["importance"] = _refine_playoffs(df)
    return df.reset_index(drop=True)


def load_cup_events(api_key: str | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if CUP_CSV.exists():
        try:
            frames.append(_events_from_fpt_like(pd.read_csv(CUP_CSV)))
        except Exception as e:
            logger.warning("fpt_cups.csv: %s", e)
    parquet = (
        Path.home()
        / "Downloads"
        / "football-ml-predictor"
        / "data"
        / "processed"
        / "fixtures.parquet"
    )
    if parquet.exists():
        try:
            fx = pd.read_parquet(parquet)
            cups = fx[
                fx["league_name"]
                .astype(str)
                .str.contains("Libertadores|Sudamericana|Copa Do Brasil", case=False, na=False)
            ].copy()
            cups = cups.rename(
                columns={
                    "kickoff": "date",
                    "round": "Round_raw",
                    "league_name": "competition",
                    "season_year": "Season",
                }
            )
            frames.append(_events_from_fpt_like(cups))
        except Exception as e:
            logger.warning("fixtures.parquet copas: %s", e)
    if not frames:
        return pd.DataFrame(
            columns=[
                "date",
                "home_team",
                "away_team",
                "round_raw",
                "competition",
                "season",
                "home_key",
                "away_key",
                "importance",
            ]
        )
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["date", "home_key", "away_key"], keep="last")
    return out.sort_values("date").reset_index(drop=True)


def download_fpt_cups(api_key: str, out_csv: Path = CUP_CSV) -> pd.DataFrame:
    from prob_ml.fpt_leagues import download_fpt_league_season, SEASONS_DEFAULT

    frames = []
    for lg in FPT_CUP_LEAGUES:
        for season in SEASONS_DEFAULT:
            try:
                df = download_fpt_league_season(lg["slug"], season, api_key)
            except Exception as e:
                logger.warning("Copa %s %s: %s", lg["slug"], season, e)
                continue
            df = df.copy()
            if "Round" in df.columns:
                df["Round_raw"] = df["Round"].astype(str)
            df["fpt_competition"] = lg["competition"]
            df["competition"] = lg["competition"]
            frames.append(df)
            logger.info("Copa %s %s → %s", lg["slug"], season, len(df))
    if not frames:
        return pd.DataFrame()
    cols: list[str] = []
    seen: set[str] = set()
    for f in frames:
        for c in f.columns:
            if c not in seen:
                cols.append(c)
                seen.add(c)
    full = pd.concat([f.reindex(columns=cols) for f in frames], ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out_csv, index=False, encoding="utf-8-sig")
    clear_context_cache()
    return full


def _long_from_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["team_key", "date", "importance"])
    h = events[["home_key", "date", "importance"]].rename(columns={"home_key": "team_key"})
    a = events[["away_key", "date", "importance"]].rename(columns={"away_key": "team_key"})
    return pd.concat([h, a], ignore_index=True).dropna(subset=["team_key", "date"])


def _dates_by_team(long: pd.DataFrame) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if long.empty:
        return out
    for team, g in long.groupby("team_key", sort=False):
        out[str(team)] = np.sort(pd.to_datetime(g["date"]).to_numpy())
    return out


def rest_days(
    team: str,
    date: Any,
    long: pd.DataFrame | None = None,
    dates_index: dict[str, np.ndarray] | None = None,
) -> float:
    dt = pd.to_datetime(date, errors="coerce", dayfirst=True)
    if pd.isna(dt):
        return DEFAULT_REST
    key = _strip(team)
    ts = None
    if dates_index is not None:
        ts = dates_index.get(key)
    elif long is not None and not long.empty:
        prev = long.loc[(long["team_key"] == key) & (long["date"] < dt), "date"]
        if prev.empty:
            return DEFAULT_REST
        delta = (dt.normalize() - pd.Timestamp(prev.max()).normalize()).days
        return float(max(1, min(int(delta), 30)))
    if ts is None or len(ts) == 0:
        return DEFAULT_REST
    target = np.datetime64(dt.to_datetime64())
    i = int(np.searchsorted(ts, target, side="left"))
    if i <= 0:
        return DEFAULT_REST
    prev = pd.Timestamp(ts[i - 1])
    delta = (dt.normalize() - prev.normalize()).days
    return float(max(1, min(int(delta), 30)))


def next_importance(
    team: str,
    date: Any,
    *,
    horizon_days: int = HORIZON_DAYS,
    events: pd.DataFrame | None = None,
) -> str:
    if events is None or events.empty:
        return "Não tem"
    dt = pd.to_datetime(date, errors="coerce", dayfirst=True)
    if pd.isna(dt):
        return "Não tem"
    key = _strip(team)
    end = dt + pd.Timedelta(days=horizon_days)
    cup = events[events["importance"] != "Não tem"]
    sub = cup[
        (cup["date"] > dt)
        & (cup["date"] <= end)
        & ((cup["home_key"] == key) | (cup["away_key"] == key))
    ]
    if sub.empty:
        return "Não tem"
    best = max(sub["importance"].tolist(), key=lambda x: IMP_RANK.get(x, 0))
    return best if best in IMP_RANK else "Não tem"


def importance_dummies(label: str) -> dict[str, float]:
    return {c: 1.0 if IMP_DUMMY_LABELS[c] == label else 0.0 for c in IMP_DUMMY_COLS}


def attach_context_to_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """home/away rest_days + importância + dummies (leakage-safe: só calendário futuro)."""
    cups = load_cup_events()
    liga = _events_from_fpt_like(matches)
    liga["importance"] = "Não tem"
    all_ev = pd.concat([cups, liga], ignore_index=True)
    all_ev = all_ev.drop_duplicates(subset=["date", "home_key", "away_key"], keep="last")
    long = _long_from_events(all_ev)
    dates_index = _dates_by_team(long)
    out = matches.copy()
    h_rest, a_rest, h_imp, a_imp = [], [], [], []
    for _, row in out.iterrows():
        dt = row.get("date")
        h = str(row.get("home_team", ""))
        a = str(row.get("away_team", ""))
        h_rest.append(rest_days(h, dt, dates_index=dates_index))
        a_rest.append(rest_days(a, dt, dates_index=dates_index))
        h_imp.append(next_importance(h, dt, events=cups))
        a_imp.append(next_importance(a, dt, events=cups))
    out["home_rest_days"] = h_rest
    out["away_rest_days"] = a_rest
    out["home_important"] = h_imp
    out["away_important"] = a_imp
    for c in IMP_DUMMY_COLS:
        out[f"home_{c}"] = [importance_dummies(x)[c] for x in h_imp]
        out[f"away_{c}"] = [importance_dummies(x)[c] for x in a_imp]
    return out


_CTX_CACHE: dict[str, Any] | None = None


def clear_context_cache() -> None:
    global _CTX_CACHE
    _CTX_CACHE = None


def _get_ctx_cache() -> dict[str, Any]:
    global _CTX_CACHE
    if _CTX_CACHE is not None:
        return _CTX_CACHE
    cups = load_cup_events()
    frames = [cups]
    liga_path = ROOT / "dados" / "fpt_matches.csv"
    if liga_path.exists():
        try:
            liga = _events_from_fpt_like(pd.read_csv(liga_path))
            liga["importance"] = "Não tem"
            frames.append(liga)
        except Exception:
            pass
    all_ev = pd.concat(frames, ignore_index=True) if frames else cups
    long = _long_from_events(all_ev)
    _CTX_CACHE = {
        "cups": cups,
        "dates_index": _dates_by_team(long),
    }
    return _CTX_CACHE


def context_for_team(team: str, date: Any) -> tuple[float, str]:
    """Descanso e importância para um time numa data (regressão / app)."""
    cache = _get_ctx_cache()
    return rest_days(team, date, dates_index=cache["dates_index"]), next_importance(
        team, date, events=cache["cups"]
    )
