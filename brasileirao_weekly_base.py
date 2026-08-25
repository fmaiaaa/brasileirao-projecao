"""
Runtime do app: só a base XLSX semanal + planilha de resultados (aba Jogos).

Arquivo de modelos: dados/brasileirao_modelos.xlsx
(também tenta a mesma planilha Google Sheets se as abas existirem).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from brasileirao_projecao_core import (
    ARQUIVO_CALENDARIO,
    Jogo,
    fator_forma_recente,
    media_pts_jogo,
    peso_forma_recente_horizonte,
    times_do_calendario,
)

_ROOT = Path(__file__).resolve().parent
_DADOS = _ROOT / "dados"
_ENTREGA = _ROOT / "artifacts" / "entrega"

# Importa nomes canônicos (scripts/ no path via sys.path do app = root)
try:
    from scripts.entrega_xlsx import (  # type: ignore
        MODELOS_XLSX_NAME,
        SHEET_CLASSIF_PROB,
        SHEET_CLASSIF_REG,
        SHEET_COEFS_REG,
        SHEET_CONTEXTO,
        SHEET_FORECASTS,
        SHEET_LEIA_ME,
        SHEET_METRICAS,
        SHEET_PROJ_PROB,
        SHEET_PROJ_REG,
    )
except Exception:  # pragma: no cover
    MODELOS_XLSX_NAME = "brasileirao_modelos.xlsx"
    SHEET_LEIA_ME = "Leia-me"
    SHEET_PROJ_REG = "Projecoes_Regressao"
    SHEET_COEFS_REG = "Coefs_Regressao"
    SHEET_CLASSIF_REG = "Classif_Regressao"
    SHEET_PROJ_PROB = "Projecoes_Prob"
    SHEET_FORECASTS = "Match_Forecasts"
    SHEET_CLASSIF_PROB = "Classif_Prob_MC"
    SHEET_METRICAS = "Metricas_Prob"
    SHEET_CONTEXTO = "Base_Contexto"


def modelos_xlsx_candidates() -> list[Path]:
    """Ordem: dados/ (junto ao calendário) → entrega latest → Downloads."""
    return [
        _DADOS / MODELOS_XLSX_NAME,
        ARQUIVO_CALENDARIO.parent / MODELOS_XLSX_NAME,
        _ENTREGA / "brasileirao_modelos_latest.xlsx",
        _ENTREGA / MODELOS_XLSX_NAME,
        Path.home() / "Downloads" / MODELOS_XLSX_NAME,
    ]


def resolve_modelos_xlsx() -> Path | None:
    for p in modelos_xlsx_candidates():
        if p.exists():
            return p
    return None


def _ler_aba_gsheets(sheet_name: str) -> pd.DataFrame | None:
    """Se a aba existir na mesma planilha de resultados, usa ela."""
    try:
        from brasileirao_gsheets import (
            credenciais_disponiveis,
            ler_aba_exata_gsheets,
            load_service_account_info,
            spreadsheet_id_brasileirao,
        )

        if not credenciais_disponiveis():
            return None
        info = load_service_account_info()
        if not info:
            return None
        return ler_aba_exata_gsheets(info, spreadsheet_id_brasileirao(), sheet_name)
    except Exception:
        return None


def load_sheet(sheet_name: str) -> pd.DataFrame | None:
    """Carrega aba: Sheets (se existir) senão XLSX local."""
    g = _ler_aba_gsheets(sheet_name)
    if g is not None and not g.empty:
        return g
    path = resolve_modelos_xlsx()
    if path is None:
        return None
    try:
        return pd.read_excel(path, sheet_name=sheet_name)
    except ValueError:
        return None
    except Exception:
        return None


def modelos_ready() -> bool:
    """Precisa ao menos das projeções de regressão e probabilístico."""
    if resolve_modelos_xlsx() is not None:
        return True
    # ou abas já coladas na planilha de resultados
    return load_sheet(SHEET_PROJ_REG) is not None and load_sheet(SHEET_PROJ_PROB) is not None


def regressao_ready() -> bool:
    return load_sheet(SHEET_PROJ_REG) is not None


def load_regressao_calendar() -> pd.DataFrame | None:
    return load_sheet(SHEET_PROJ_REG)


def load_regressao_coefs() -> pd.DataFrame | None:
    df = load_sheet(SHEET_COEFS_REG)
    if df is None or df.empty:
        return None
    df = df.copy()
    # Sheets (locale BR) devolve "0,8888" → Streamlit NumberColumn vira 8888
    r2_col = None
    for c in df.columns:
        cl = str(c).strip().lower().replace("²", "2")
        if cl in {"r2", "r²"} or cl.replace(" ", "") == "r2":
            r2_col = c
            break
    if r2_col is not None:
        def _r2(v):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return None
            s = str(v).strip().replace(",", ".")
            if s == "" or s.lower() in {"nan", "none", "-", "—"}:
                return None
            try:
                return round(float(s), 4)
            except (TypeError, ValueError):
                return None

        df[r2_col] = df[r2_col].map(_r2)
    return df


def load_prob_projecoes() -> pd.DataFrame | None:
    return load_sheet(SHEET_PROJ_PROB)


def load_prob_forecasts() -> pd.DataFrame | None:
    return load_sheet(SHEET_FORECASTS)


def load_prob_standings() -> pd.DataFrame | None:
    return load_sheet(SHEET_CLASSIF_PROB)


def load_base_contexto() -> pd.DataFrame | None:
    return load_sheet(SHEET_CONTEXTO)


def anos_disponiveis_estatisticas(*, ano_atual: int = 2026) -> list[int]:
    """Anos com Série A na Base_Contexto + ano atual do calendário."""
    anos: set[int] = {int(ano_atual)}
    ctx = load_base_contexto()
    if ctx is None or ctx.empty or "season" not in ctx.columns:
        return sorted(anos, reverse=True)
    df = ctx.copy()
    if "competition" in df.columns:
        comp = df["competition"].astype(str).str.lower()
        df = df[comp.str.contains("serie_a|serie a|betano", regex=True, na=False)]
    for s in pd.to_numeric(df["season"], errors="coerce").dropna().unique():
        anos.add(int(s))
    return sorted(anos, reverse=True)


def jogos_serie_a_ano(ano: int) -> list[Jogo]:
    """
    Converte jogos da Série A (Base_Contexto) em list[Jogo] para estatísticas.
    Só inclui partidas com placar (home_goals/away_goals).
    """
    ctx = load_base_contexto()
    if ctx is None or ctx.empty:
        return []
    df = ctx.copy()
    if "competition" in df.columns:
        comp = df["competition"].astype(str).str.lower()
        df = df[comp.str.contains("serie_a|serie a|betano", regex=True, na=False)]
    df = df[pd.to_numeric(df.get("season"), errors="coerce") == int(ano)]
    played = df["home_goals"].notna() & df["away_goals"].notna()
    df = df.loc[played].copy()
    if df.empty:
        return []
    if "date" in df.columns:
        df = df.sort_values(["date", "home_team"], kind="mergesort")
    jogos: list[Jogo] = []
    for _, row in df.iterrows():
        try:
            r = int(float(row["round"])) if pd.notna(row.get("round")) else 1
        except (TypeError, ValueError):
            r = 1
        try:
            hg, ag = int(row["home_goals"]), int(row["away_goals"])
        except (TypeError, ValueError):
            continue
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
                placar=f"{hg} x {ag}",
                est="serie_a",
            )
        )
    return jogos


def load_prob_metricas() -> pd.DataFrame | None:
    return load_sheet(SHEET_METRICAS)


def weekly_meta_caption() -> str:
    leia = load_sheet(SHEET_LEIA_ME)
    if leia is not None and not leia.empty and {"campo", "valor"} <= set(leia.columns):
        by = {str(r["campo"]): str(r["valor"]) for _, r in leia.iterrows()}
        gerado = by.get("gerado_em") or by.get("saved_at") or "—"
        champ = by.get("champion", "—")
        src = resolve_modelos_xlsx()
        where = f" · {src.name}" if src else " · planilha Sheets"
        return f"Base modelos de {gerado} · champion={champ}{where}"
    path = resolve_modelos_xlsx()
    if path is not None:
        return (
            "Base modelos (XLSX) de "
            f"{datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec='seconds')} "
            f"· {path}"
        )
    return "Base modelos ausente — coloque brasileirao_modelos.xlsx junto aos resultados."


def estatisticas_contexto_por_time(ano: int) -> pd.DataFrame:
    """
    Estatísticas extras da Base_Contexto (descanso, jogos importantes, xG)
    para a Série A do ano — sem leakage (só jogos com placar da temporada).
    """
    ctx = load_base_contexto()
    if ctx is None or ctx.empty:
        return pd.DataFrame()
    df = ctx.copy()
    if "competition" in df.columns:
        comp = df["competition"].astype(str).str.lower()
        df = df[comp.str.contains("serie_a|serie a|betano", regex=True, na=False)]
    df = df[pd.to_numeric(df.get("season"), errors="coerce") == int(ano)]
    # placar: aceita numérico ou string (Sheets)
    hg = pd.to_numeric(df.get("home_goals"), errors="coerce")
    ag = pd.to_numeric(df.get("away_goals"), errors="coerce")
    df = df.loc[hg.notna() & ag.notna()].copy()
    if df.empty:
        return pd.DataFrame()

    def _num(val: Any) -> float | None:
        if val is None:
            return None
        try:
            if isinstance(val, float) and pd.isna(val):
                return None
        except Exception:
            pass
        s = str(val).strip()
        if s == "" or s.lower() in {"nan", "none", "-", "—"}:
            return None
        s = s.replace(",", ".")
        try:
            return float(s)
        except (TypeError, ValueError):
            return None

    rows: dict[str, dict[str, float]] = {}

    def _acc(team: str) -> dict[str, float]:
        if team not in rows:
            rows[team] = {
                "n": 0.0,
                "rest_sum": 0.0,
                "rest_n": 0.0,
                "imp_n": 0.0,
                "xg_for": 0.0,
                "xg_against": 0.0,
                "xg_n": 0.0,
            }
        return rows[team]

    for _, r in df.iterrows():
        h, a = str(r["home_team"]).strip(), str(r["away_team"]).strip()
        if not h or h.lower() == "nan" or not a or a.lower() == "nan":
            continue
        for team, rest_col, imp_col, xg_for_col, xg_ag_col in (
            (h, "home_rest_days", "home_important", "home_xg", "away_xg"),
            (a, "away_rest_days", "away_important", "away_xg", "home_xg"),
        ):
            acc = _acc(team)
            acc["n"] += 1
            rest = _num(r.get(rest_col))
            if rest is not None:
                acc["rest_sum"] += rest
                acc["rest_n"] += 1
            imp = str(r.get(imp_col) or "Não tem").strip()
            if imp and imp != "Não tem" and imp.lower() != "nan":
                acc["imp_n"] += 1
            xgf = _num(r.get(xg_for_col))
            xga = _num(r.get(xg_ag_col))
            if xgf is not None and xga is not None:
                acc["xg_for"] += xgf
                acc["xg_against"] += xga
                acc["xg_n"] += 1

    out_rows = []
    for team, acc in rows.items():
        out_rows.append(
            {
                "Time": team,
                "Média dias de descanso": round(
                    acc["rest_sum"] / acc["rest_n"], 2
                )
                if acc["rest_n"]
                else None,
                "% jogos c/ importante à frente": round(
                    100.0 * acc["imp_n"] / acc["n"], 1
                )
                if acc["n"]
                else 0.0,
                "Média xG marcados": round(acc["xg_for"] / acc["xg_n"], 3)
                if acc["xg_n"]
                else None,
                "Média xG sofridos": round(acc["xg_against"] / acc["xg_n"], 3)
                if acc["xg_n"]
                else None,
                "Média xG marcados/Média xG sofridos": round(
                    (acc["xg_for"] / acc["xg_n"]) / (acc["xg_against"] / acc["xg_n"]),
                    3,
                )
                if acc["xg_n"] and (acc["xg_against"] / acc["xg_n"]) > 1e-9
                else None,
            }
        )
    return pd.DataFrame(out_rows)


def enriquecer_stats_com_contexto(
    df_stats: pd.DataFrame, ano: int
) -> pd.DataFrame:
    try:
        extra = estatisticas_contexto_por_time(int(ano))
    except Exception:
        return df_stats
    if extra.empty or df_stats is None or df_stats.empty:
        return df_stats
    from brasileirao_multi_liga import align_team_to_calendar

    cal = set(df_stats["Time"].astype(str))
    extra = extra.copy()
    extra["Time"] = extra["Time"].map(lambda t: align_team_to_calendar(str(t), cal))
    # evita linhas duplicadas (ex.: Flamengo + Flamengo RJ → mesmo Time)
    num_cols = [
        c
        for c in extra.columns
        if c != "Time" and pd.api.types.is_numeric_dtype(extra[c])
    ]
    if num_cols:
        extra = (
            extra.groupby("Time", as_index=False)[num_cols]
            .mean(numeric_only=True)
        )
    else:
        extra = extra.drop_duplicates(subset=["Time"], keep="first")

    base = df_stats.drop_duplicates(subset=["Time"], keep="first")
    merged = base.merge(extra, on="Time", how="left")
    return merged.drop_duplicates(subset=["Time"], keep="first")


def gap_fill_media_apenas_faltantes(
    jogos: list[Jogo],
    *,
    r_ini: int,
    r_fim: int,
) -> list[dict[str, Any]]:
    faltantes = [j for j in jogos if (not j.jogado) and j.proj_pm is None]
    if not faltantes:
        return []

    times = times_do_calendario(jogos)
    medias = {
        t: media_pts_jogo(jogos, t, r_ini, r_fim, "mandante_visitante") for t in times
    }
    fatores = {t: fator_forma_recente(jogos, t, r_ini, r_fim) for t in times}
    ult_r_real = max((j.r for j in jogos if j.jogado), default=r_fim)
    log_rows: list[dict[str, Any]] = []

    for j in faltantes:
        w = peso_forma_recente_horizonte(max(1, j.r - ult_r_real))
        fat_m = w * fatores[j.mand] + (1.0 - w)
        fat_v = w * fatores[j.vis] + (1.0 - w)
        mm = medias[j.mand]
        mv = medias[j.vis]
        pm = max(mm.get("casa", mm["geral"]) * fat_m, 0.0)
        pv = max(mv.get("fora", mv["geral"]) * fat_v, 0.0)
        j.proj_pm, j.proj_pv = pm, pv
        j.origem = "gap_fill/media_calendario"
        log_rows.append(
            {
                "Rodada": j.r,
                "Mandante": j.mand,
                "Visitante": j.vis,
                "Proj": f"{pm:.2f} / {pv:.2f}",
                "Fonte": "gap_fill/media",
            }
        )
    return log_rows


def aplicar_projecoes_csv_com_gap(
    jogos: list[Jogo],
    calendar_csv: pd.DataFrame | None,
    *,
    r_ini: int = 1,
    r_fim: int = 38,
    origem: str = "weekly_xlsx",
) -> tuple[list[Jogo], pd.DataFrame]:
    from prob_ml.integration import aplicar_projecoes_de_csv

    if calendar_csv is not None and not calendar_csv.empty:
        jogos_out, df_log = aplicar_projecoes_de_csv(jogos, calendar_csv)
        for j in jogos_out:
            if j.proj_pm is not None and not j.jogado:
                j.origem = origem
    else:
        jogos_out = [Jogo(**j.__dict__) for j in jogos]
        df_log = pd.DataFrame()

    gap_rows = gap_fill_media_apenas_faltantes(jogos_out, r_ini=r_ini, r_fim=r_fim)
    rows: list[dict[str, Any]] = []
    if not df_log.empty:
        for r in df_log.to_dict(orient="records"):
            r.setdefault("Fonte", origem)
            rows.append(r)
    rows.extend(gap_rows)
    return jogos_out, pd.DataFrame(rows)
