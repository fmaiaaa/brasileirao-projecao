"""Download e consolidação multi-campeonato FutPythonTrader (FPT)."""
from __future__ import annotations

import io
import logging
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Campeonatos usados no treino de regressão / probabilístico.
# Médias e repetir 1º turno NÃO usam estas bases — só o calendário do Brasileirão.
FPT_LEAGUES_DEFAULT: list[dict[str, str]] = [
    {"slug": "brazil/serie-a-betano", "competition": "serie_a"},
    {"slug": "brazil/serie-b", "competition": "serie_b"},
    {"slug": "brazil/serie-c", "competition": "serie_c"},
    {"slug": "brazil/serie-d", "competition": "serie_d"},
    {"slug": "brazil/brasileiro-women", "competition": "brasileiro_women"},
]

FPT_URL = (
    "https://futpythontrader.com.br/api/download/{slug}/{season}?api_key={key}"
)
SEASONS_DEFAULT = [2021, 2022, 2023, 2024, 2025, 2026]


def _parse_round_series(raw: pd.Series) -> pd.Series:
    extracted = raw.astype(str).str.extract(r"(\d+)", expand=False)
    out = pd.to_numeric(extracted, errors="coerce")
    return out


def _synthesize_rounds_by_date(dates: pd.Series) -> pd.Series:
    """Quando Round não é numérico (Série C/D), usa ordem cronológica de datas."""
    d = pd.to_datetime(dates, errors="coerce", dayfirst=True)
    uniq = sorted(x for x in d.dropna().unique())
    mp = {u: i + 1 for i, u in enumerate(uniq)}
    return d.map(mp)


def download_fpt_league_season(slug: str, season: int, api_key: str) -> pd.DataFrame:
    url = FPT_URL.format(slug=slug, season=season, key=api_key)
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    if b"<html" in r.content[:200].lower():
        raise RuntimeError(f"Resposta HTML em {slug}/{season}")
    df = pd.read_csv(io.BytesIO(r.content))
    if "Season" not in df.columns:
        df["Season"] = season
    return df


def download_fpt_multi(
    api_key: str,
    out_csv: Path,
    *,
    leagues: list[dict[str, str]] | None = None,
    seasons: list[int] | None = None,
) -> pd.DataFrame:
    """Baixa várias ligas/temporadas e grava CSV consolidado."""
    leagues = leagues or FPT_LEAGUES_DEFAULT
    seasons = seasons or SEASONS_DEFAULT
    frames: list[pd.DataFrame] = []
    for lg in leagues:
        slug = lg["slug"]
        comp = lg["competition"]
        for season in seasons:
            try:
                df = download_fpt_league_season(slug, season, api_key)
            except Exception as e:
                logger.warning("Falha %s %s: %s", slug, season, e)
                continue
            df = df.copy()
            if "Round" in df.columns:
                df["Round_raw"] = df["Round"].astype(str)
            df["fpt_competition"] = comp
            df["competition"] = comp
            df["fpt_slug"] = slug
            if "Season" not in df.columns:
                df["Season"] = season
            # Round numérico para ligas; copas guardam o rótulo em Round_raw
            if "Round" in df.columns:
                rnd = _parse_round_series(df["Round"])
                if rnd.notna().mean() < 0.5 and "Date" in df.columns:
                    rnd = _synthesize_rounds_by_date(df["Date"])
                df["Round"] = rnd
            logger.info("%s %s → %s linhas", slug, season, len(df))
            frames.append(df)

    if not frames:
        raise RuntimeError("Nenhuma liga/temporada baixada da FPT")

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
    logger.info("Multi-liga salva em %s (%s linhas)", out_csv, len(full))
    return full


def leagues_from_config(cfg: dict[str, Any] | None) -> list[dict[str, str]]:
    data = (cfg or {}).get("data") or {}
    raw = data.get("fpt_leagues")
    if not raw:
        return list(FPT_LEAGUES_DEFAULT)
    out: list[dict[str, str]] = []
    for item in raw:
        if isinstance(item, str):
            slug = item
            comp = slug.rstrip("/").split("/")[-1].replace("-", "_")
            out.append({"slug": slug, "competition": comp})
        elif isinstance(item, dict) and item.get("slug"):
            out.append(
                {
                    "slug": str(item["slug"]),
                    "competition": str(item.get("competition") or item["slug"]),
                }
            )
    return out or list(FPT_LEAGUES_DEFAULT)
