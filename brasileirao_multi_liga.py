"""
Bases multi-campeonato para treino da regressão (não usadas em médias / 1º turno).

Converte o CSV FPT consolidado em blocos list[Jogo] por (competition, season).
"""
from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

from brasileirao_projecao_core import Jogo
from prob_ml.config import load_config
from prob_ml.data import LocalFileDataSource
from recency import filter_matches_dataframe, load_recency_settings

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent

# Outros campeonatos no painel da regressão (além do calendário atual).
# Probabilístico usa a base FPT completa (todas as ligas do config).
REGRESSAO_COMPETITIONS = {"serie_a", "serie_b", "brasileiro_women"}


def normalize_competition(raw: str) -> str:
    s = str(raw).strip().lower()
    if "women" in s or "femin" in s:
        return "brasileiro_women"
    if "serie d" in s or s == "serie_d":
        return "serie_d"
    if "serie c" in s or s == "serie_c":
        return "serie_c"
    if "serie b" in s or s == "serie_b":
        return "serie_b"
    if "serie a" in s or "betano" in s or s == "serie_a":
        return "serie_a"
    return s.replace(" ", "_").replace("-", "_")


def _strip_accents(s: str) -> str:
    return "".join(
        c
        for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )


def _norm_team(s: str) -> str:
    t = _strip_accents(str(s).strip().lower())
    t = re.sub(r"\s+", " ", t)
    return t


# Calendário app → possíveis rótulos FPT
_CAL_TO_FPT_HINTS: dict[str, list[str]] = {
    "Flamengo": ["Flamengo RJ", "Flamengo"],
    "Botafogo": ["Botafogo RJ", "Botafogo"],
    "Atlético-MG": ["Atletico-MG", "Atlético-MG", "Atletico Mineiro"],
    "Athletico-PR": ["Athletico-PR", "Athletico Paranaense"],
    "São Paulo": ["Sao Paulo", "São Paulo"],
    "Grêmio": ["Gremio", "Grêmio"],
    "Cuiabá": ["Cuiaba", "Cuiabá"],
    "Vitória": ["Vitoria", "Vitória"],
    "Ceará": ["Ceara", "Ceará"],
    "Goiás": ["Goias", "Goiás"],
    "América-MG": ["America-MG", "America MG"],
    "Chapecoense": ["Chapecoense-SC", "Chapecoense"],
}


def align_team_to_calendar(name: str, calendar_teams: set[str]) -> str:
    """Se o time FPT corresponder a um do calendário, devolve o nome do calendário."""
    if name in calendar_teams:
        return name
    n = _norm_team(name)
    for cal in calendar_teams:
        if _norm_team(cal) == n:
            return cal
        for hint in _CAL_TO_FPT_HINTS.get(cal, []):
            if _norm_team(hint) == n:
                return cal
        # sufixos RJ/MG/PR/SC
        if n.startswith(_norm_team(cal)) or _norm_team(cal).startswith(n):
            return cal
    return name


def matches_df_to_jogo_blocks(
    matches: pd.DataFrame,
    *,
    exclude_serie_a_season: int | None = 2026,
    calendar_teams: set[str] | None = None,
    cfg: dict | None = None,
) -> list[list[Jogo]]:
    """
    Cada bloco = uma temporada de um campeonato (força/forma calculadas dentro do bloco).
    Exclui Série A da temporada atual (já coberta pelo calendário do app).
    Histórico: últimas 38 rodadas da Série A + Série B para times promovidos.
    """
    rcfg = load_recency_settings(cfg)
    df = filter_matches_dataframe(
        matches,
        history_rounds=int(rcfg["history_rounds"]),
        calendar_teams=calendar_teams,
    )
    df = df.copy()
    if "home_goals" not in df.columns:
        raise ValueError("matches sem home_goals")
    played = df["home_goals"].notna() & df["away_goals"].notna()
    df = df.loc[played].copy()
    if "competition" not in df.columns:
        df["competition"] = "serie_a"
    df["competition"] = df["competition"].map(normalize_competition)
    df = df[df["competition"].isin(REGRESSAO_COMPETITIONS)].copy()
    if df.empty:
        logger.warning("Nenhum jogo nas competições de regressão %s", REGRESSAO_COMPETITIONS)
        return []
    if "season" not in df.columns:
        df["season"] = 0
    if "round" not in df.columns or df["round"].isna().all():
        df["round"] = 1

    blocks: list[list[Jogo]] = []
    for (comp, season), g in df.groupby(["competition", "season"], dropna=False):
        if (
            exclude_serie_a_season is not None
            and str(comp).lower() in {"serie_a", "serie-a-betano", "serie-a"}
            and int(season) == int(exclude_serie_a_season)
        ):
            continue
        jogos: list[Jogo] = []
        g2 = g.sort_values(["date", "home_team"], kind="mergesort")
        for _, row in g2.iterrows():
            try:
                r = int(row["round"]) if pd.notna(row["round"]) else 1
            except (TypeError, ValueError):
                r = 1
            hg, ag = int(row["home_goals"]), int(row["away_goals"])
            data = ""
            if pd.notna(row.get("date")):
                data = str(pd.Timestamp(row["date"]))[:10]
            jogos.append(
                Jogo(
                    r=r,
                    data=data,
                    hora="",
                    mand=str(row["home_team"]).strip(),
                    vis=str(row["away_team"]).strip(),
                    placar=f"{hg}x{ag}",
                    est=str(comp),
                )
            )
        if len(jogos) >= 20:
            blocks.append(jogos)
    logger.info(
        "Blocos multi-liga para regressão: %s (jogos=%s)",
        len(blocks),
        sum(len(b) for b in blocks),
    )
    return blocks


def carregar_blocos_treino_regressao(
    cfg: dict[str, Any] | None = None,
    *,
    exclude_serie_a_season: int | None = 2026,
    calendar_teams: set[str] | None = None,
) -> list[list[Jogo]]:
    cfg = cfg or load_config()
    path = ROOT / cfg.get("data", {}).get("local_path", "dados/fpt_matches.csv")
    if not path.exists():
        logger.warning("Base FPT ausente (%s) — regressão só com calendário", path)
        return []
    matches, _ = LocalFileDataSource(path).load_canonical()
    return matches_df_to_jogo_blocks(
        matches,
        exclude_serie_a_season=exclude_serie_a_season,
        calendar_teams=calendar_teams,
        cfg=cfg,
    )


def remap_block_teams(bloco: list[Jogo], calendar_teams: set[str]) -> list[Jogo]:
    """Alinha nomes FPT aos do calendário quando há correspondência."""
    out: list[Jogo] = []
    for j in bloco:
        out.append(
            Jogo(
                r=j.r,
                data=j.data,
                hora=j.hora,
                mand=align_team_to_calendar(j.mand, calendar_teams),
                vis=align_team_to_calendar(j.vis, calendar_teams),
                placar=j.placar,
                est=j.est,
            )
        )
    return out
