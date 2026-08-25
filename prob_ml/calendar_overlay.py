"""
Mescla resultados frescos do calendário (Google Sheets / xlsx) na base FPT.

A FPT costuma ter ~1 rodada de atraso. A planilha compartilhada é a fonte
de verdade para placares do Brasileirão atual nos modos Regressão e Probabilístico.
Médias e Repetir 1º turno já usam só o calendário.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any

import numpy as np
import pandas as pd

from brasileirao_projecao_core import Jogo, parse_placar

logger = logging.getLogger(__name__)


def _strip(s: str) -> str:
    t = "".join(
        c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c)
    )
    return re.sub(r"\s+", " ", t.strip().lower())


def _teams_match(a: str, b: str) -> bool:
    na, nb = _strip(a), _strip(b)
    if na == nb:
        return True
    if na.startswith(nb) or nb.startswith(na):
        return True
    # remove sufixos comuns FPT
    for suf in (" rj", " mg", " pr", " sc", " sp"):
        if na.endswith(suf):
            na = na[: -len(suf)].strip()
        if nb.endswith(suf):
            nb = nb[: -len(suf)].strip()
    return na == nb


def overlay_fpt_with_calendar(
    matches: pd.DataFrame,
    jogos: list[Jogo],
    *,
    season: int = 2026,
    competition: str = "serie_a",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Atualiza/insere placares da Série A (season) a partir do calendário.

    - Se o jogo existe na FPT sem placar (ou com placar) → sobrescreve com Sheets
    - Se o jogo só existe no calendário → acrescenta linha canônica mínima
    """
    out = matches.copy()
    report: dict[str, Any] = {
        "n_calendar_played": 0,
        "n_updated": 0,
        "n_inserted": 0,
        "n_unchanged": 0,
        "season": season,
    }

    played = [j for j in jogos if j.jogado]
    report["n_calendar_played"] = len(played)
    if not played:
        return out, report

    if "competition" in out.columns:
        comp = out["competition"].astype(str).str.lower()
        mask_sa = comp.eq(competition) | comp.str.contains("serie_a|serie a|betano", regex=True)
    else:
        mask_sa = pd.Series(True, index=out.index)
    if "season" in out.columns:
        mask_sa = mask_sa & (pd.to_numeric(out["season"], errors="coerce") == season)

    sa = out.loc[mask_sa].copy()
    used_idx: set[Any] = set()

    def _find_row(j: Jogo) -> Any | None:
        cand = sa
        if "round" in cand.columns:
            rmask = pd.to_numeric(cand["round"], errors="coerce") == int(j.r)
            sub = cand.loc[rmask]
            if sub.empty:
                sub = cand
        else:
            sub = cand
        for idx, row in sub.iterrows():
            if idx in used_idx:
                continue
            if _teams_match(str(row["home_team"]), j.mand) and _teams_match(
                str(row["away_team"]), j.vis
            ):
                return idx
        # tenta sem filtro de rodada
        for idx, row in cand.iterrows():
            if idx in used_idx:
                continue
            if _teams_match(str(row["home_team"]), j.mand) and _teams_match(
                str(row["away_team"]), j.vis
            ):
                return idx
        return None

    new_rows: list[dict[str, Any]] = []
    for j in played:
        sc = parse_placar(j.placar)
        if sc is None:
            continue
        hg, ag = sc
        idx = _find_row(j)
        if idx is not None:
            used_idx.add(idx)
            old_h = out.at[idx, "home_goals"] if "home_goals" in out.columns else np.nan
            old_a = out.at[idx, "away_goals"] if "away_goals" in out.columns else np.nan
            same = (
                pd.notna(old_h)
                and pd.notna(old_a)
                and int(old_h) == hg
                and int(old_a) == ag
            )
            if same:
                report["n_unchanged"] += 1
            else:
                out.at[idx, "home_goals"] = hg
                out.at[idx, "away_goals"] = ag
                report["n_updated"] += 1
        else:
            new_rows.append(
                {
                    "match_id": f"cal-{season}-{j.r}-{j.mand}-{j.vis}",
                    "season": season,
                    "competition": competition,
                    "date": pd.to_datetime(j.data, errors="coerce") if j.data else pd.NaT,
                    "kickoff_time": j.hora or "",
                    "round": j.r,
                    "home_team": j.mand,
                    "away_team": j.vis,
                    "home_goals": hg,
                    "away_goals": ag,
                }
            )
            report["n_inserted"] += 1

    if new_rows:
        add = pd.DataFrame(new_rows)
        # alinha colunas
        for c in out.columns:
            if c not in add.columns:
                add[c] = np.nan
        add = add.reindex(columns=list(out.columns) + [c for c in add.columns if c not in out.columns])
        out = pd.concat([out, add[out.columns]], ignore_index=True)

    logger.info(
        "Overlay calendário→FPT: updated=%s inserted=%s unchanged=%s (played=%s)",
        report["n_updated"],
        report["n_inserted"],
        report["n_unchanged"],
        report["n_calendar_played"],
    )
    return out, report


def jogos_pendentes(jogos: list[Jogo]) -> list[Jogo]:
    return [j for j in jogos if not j.jogado]
