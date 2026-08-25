"""Carrega a base canônica: Drive (se configurado/acessível) → local → calendário."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from brasileirao_projecao_core import Jogo
from prob_ml.config import load_config
from prob_ml.data import (
    GoogleDriveDataSource,
    LocalFileDataSource,
    matches_from_calendar_jogos,
)
from prob_ml.pipeline import FittedBundle, train_pipeline

logger = logging.getLogger(__name__)
_ROOT = Path(__file__).resolve().parent.parent


def load_matches_for_training(
    jogos: list[Jogo] | None = None,
    cfg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Ordem:
    1. Google Drive (se source=google_drive ou se local falhar e houver file_id)
    2. Arquivo local (dados/fpt_matches.*)
    3. Fallback: calendário do app

    Depois, se ``jogos`` (Sheets/xlsx) for passado, sobrescreve placares da
    Série A atual — a planilha é mais rápida que a FPT (~1 rodada de atraso).
    """
    cfg = cfg or load_config()
    data_cfg = cfg.get("data", {})
    report: dict[str, Any] = {"ok": False}

    file_id = (data_cfg.get("google_drive_file_id") or "").strip()
    file_url = (data_cfg.get("google_drive_file_url") or "").strip()
    source = str(data_cfg.get("source", "local")).lower()
    local_path = _ROOT / data_cfg.get("local_path", "dados/fpt_matches.csv")

    matches: pd.DataFrame | None = None

    # Preferência Drive quando source=google_drive
    if source == "google_drive" and (file_id or file_url):
        try:
            ds = GoogleDriveDataSource(
                file_id=file_id or None,
                file_url=file_url or None,
                cache_dir=_ROOT / data_cfg.get("cache_dir", "artifacts/prob_ml/cache"),
            )
            matches, report = ds.load_canonical()
            report["ok"] = True
        except Exception as e:
            logger.warning("Drive indisponível (%s); tentando local", e)
            report["drive_error"] = f"{type(e).__name__}: {e}"

    if matches is None and local_path.exists():
        matches, report = LocalFileDataSource(local_path).load_canonical()
        report["ok"] = True
        report["fallback"] = "local_file"
        if file_id or file_url:
            report["drive_configured"] = True
            report["drive_file_id"] = file_id or None

    if matches is None and (file_id or file_url):
        try:
            ds = GoogleDriveDataSource(
                file_id=file_id or None,
                file_url=file_url or None,
                cache_dir=_ROOT / data_cfg.get("cache_dir", "artifacts/prob_ml/cache"),
            )
            matches, report = ds.load_canonical()
            report["ok"] = True
        except Exception as e:
            report["drive_error"] = f"{type(e).__name__}: {e}"

    if matches is None and jogos is not None:
        matches = matches_from_calendar_jogos(jogos)
        report = {
            "ok": True,
            "fallback": "calendar",
            "n_rows": len(matches),
            "warning": "Base FPT ausente; usando calendário do app",
        }

    if matches is None:
        raise FileNotFoundError(
            "Nenhuma base disponível. Configure Drive (compartilhe com a SA) "
            "ou coloque dados/fpt_matches.csv"
        )

    if jogos:
        from prob_ml.calendar_overlay import overlay_fpt_with_calendar

        matches, ov = overlay_fpt_with_calendar(matches, jogos)
        report["calendar_overlay"] = ov

    return matches, report


def fit_from_jogos(
    jogos: list[Jogo],
    *,
    run_backtest: bool = True,
    cfg: dict[str, Any] | None = None,
    progress=None,
) -> FittedBundle:
    """Treina somente quando chamado (modo selecionado no app)."""
    cfg = cfg or load_config()
    matches, report = load_matches_for_training(jogos, cfg)
    bundle = train_pipeline(
        matches, cfg, run_backtest=run_backtest, progress=progress
    )
    src = report.get("source") or report.get("fallback") or "mapped"
    bundle.status.notes.append(f"data_source={src}")
    if report.get("fingerprint"):
        bundle.status.dataset_fingerprint = report["fingerprint"]
    if report.get("drive_error"):
        bundle.status.notes.append(f"drive_error={report['drive_error']}")
    ov = report.get("calendar_overlay") or {}
    if ov:
        bundle.status.notes.append(
            f"calendar_overlay updated={ov.get('n_updated')} "
            f"inserted={ov.get('n_inserted')}"
        )
    return bundle


def _safe_float(val, default: float | None = None) -> float | None:
    if val is None:
        return default
    try:
        import pandas as _pd

        if isinstance(val, float) and _pd.isna(val):
            return default
    except Exception:
        pass
    s = str(val).strip()
    if s == "" or s.lower() in {"nan", "none", "-", "—", "null"}:
        return default
    s = s.replace(",", ".")
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def aplicar_projecoes_de_csv(
    jogos: list[Jogo],
    calendar_csv: pd.DataFrame,
) -> tuple[list[Jogo], pd.DataFrame]:
    """Aplica xPts de CSV/Sheets offline aos jogos pendentes do calendário."""
    jogos = [Jogo(**j.__dict__) for j in jogos]
    df = calendar_csv.copy()
    # normaliza nomes de colunas
    colmap = {c.lower().strip(): c for c in df.columns}

    def col(*names: str) -> str | None:
        for n in names:
            if n.lower() in colmap:
                return colmap[n.lower()]
            if n in df.columns:
                return n
        return None

    c_r = col("Rodada", "round")
    c_m = col("Mandante", "home_team")
    c_v = col("Visitante", "away_team")
    c_proj = col("Proj")
    c_xph = col("xpts_home")
    c_xpv = col("xpts_away")
    c_1x2 = col("1X2")
    c_lam = col("λ", "lambda")

    by_key: dict[tuple, dict] = {}
    for _, row in df.iterrows():
        r = -1
        if c_r:
            rv = _safe_float(row[c_r])
            if rv is not None:
                r = int(rv)
        m = str(row[c_m]).strip() if c_m else ""
        v = str(row[c_v]).strip() if c_v else ""
        if not m or not v or m.lower() == "nan":
            continue
        by_key[(m, v, r)] = row.to_dict()
        by_key[(m, v, -1)] = row.to_dict()

    log_rows = []
    for j in jogos:
        if j.jogado:
            continue
        row = by_key.get((j.mand, j.vis, j.r)) or by_key.get((j.mand, j.vis, -1))
        if row is None:
            continue
        pm = pv = None
        if c_xph and c_xpv:
            pm = _safe_float(row.get(c_xph))
            pv = _safe_float(row.get(c_xpv))
        if (pm is None or pv is None) and c_proj and row.get(c_proj):
            parts = str(row[c_proj]).replace(",", ".").split("/")
            if len(parts) == 2:
                pm = _safe_float(parts[0].strip(), pm)
                pv = _safe_float(parts[1].strip(), pv)
        if pm is None or pv is None:
            continue
        j.proj_pm, j.proj_pv = float(pm), float(pv)
        j.origem = "prob_ml/offline_csv"
        log_rows.append(
            {
                "Rodada": j.r,
                "Mandante": j.mand,
                "Visitante": j.vis,
                "Proj": f"{j.proj_pm:.2f} / {j.proj_pv:.2f}",
                "λ": row.get(c_lam, "") if c_lam else "",
                "1X2": row.get(c_1x2, "") if c_1x2 else "",
            }
        )
    return jogos, pd.DataFrame(log_rows)


def refresh_standings_from_calendar(
    jogos: list[Jogo],
    bundle: FittedBundle,
    *,
    cfg: dict[str, Any] | None = None,
) -> tuple[list[dict], Any]:
    """
    Recalcula previsões + Monte Carlo só com jogos ainda pendentes no calendário.
    Use mid-week quando a Sheets ganha placares novos e o CSV semanal ficou velho.
    """
    from brasileirao_projecao_core import stats_acumuladas_ate, times_do_calendario
    from prob_ml.config import budget_n_sims, load_config
    from prob_ml.offline import forecast_calendar_games
    from prob_ml.simulation import result_to_frame, simulate_season

    cfg = cfg or load_config()
    forecasts = forecast_calendar_games(bundle, jogos)
    teams = times_do_calendario(jogos)
    current = {t: {"points": 0.0, "wins": 0.0, "gd": 0.0, "gf": 0.0} for t in teams}
    for t in teams:
        st = stats_acumuladas_ate(jogos, t, 38, so_realizados=True)
        current[t] = {
            "points": float(st.pts),
            "wins": float(st.vit),
            "gd": float(st.sg),
            "gf": float(st.gf),
        }
    fixtures = [
        (p["home_team"], p["away_team"], p["dist"]) for p in forecasts if "dist" in p
    ]
    sim_df = None
    if fixtures:
        res = simulate_season(
            teams, current, fixtures, n_sims=budget_n_sims(cfg), seed=42
        )
        sim_df = result_to_frame(res)
    preds = [{k: v for k, v in p.items() if k != "dist"} for p in forecasts]
    return preds, sim_df


def aplicar_projecoes_probabilisticas(
    jogos: list[Jogo],
    bundle: FittedBundle,
) -> tuple[list[Jogo], pd.DataFrame, list[dict], Any]:
    """
    Preenche proj_pm/proj_pv com expected points da distribuição de placar.
    Retorna jogos, log, previsões detalhadas, resultado MC (ou None).
    """
    from prob_ml.pipeline import predict_fixtures, simulate_from_bundle

    jogos = [Jogo(**j.__dict__) for j in jogos]
    preds = predict_fixtures(bundle)
    by_key = {
        (str(p["home_team"]), str(p["away_team"]), int(p["round"] or -1)): p
        for p in preds
    }
    log_rows = []
    for j in jogos:
        if j.jogado:
            continue
        p = by_key.get((j.mand, j.vis, j.r))
        if p is None:
            for k, v in by_key.items():
                if k[0] == j.mand and k[1] == j.vis:
                    p = v
                    break
        if p is None:
            continue
        j.proj_pm = float(p["xpts_home"])
        j.proj_pv = float(p["xpts_away"])
        j.proj_gm = int(round(p["xg_home"]))
        j.proj_gv = int(round(p["xg_away"]))
        j.origem = (
            f"prob_ml/{p.get('champion', 'ensemble')} "
            f"P(H/D/A)={p['p_home']:.0%}/{p['p_draw']:.0%}/{p['p_away']:.0%}"
        )
        log_rows.append(
            {
                "Rodada": j.r,
                "Mandante": j.mand,
                "Visitante": j.vis,
                "Proj": f"{j.proj_pm:.2f} / {j.proj_pv:.2f}",
                "λ": f"{p['xg_home']:.2f} x {p['xg_away']:.2f}",
                "1X2": f"{p['p_home']:.0%} / {p['p_draw']:.0%} / {p['p_away']:.0%}",
            }
        )

    sim_df = None
    try:
        _, sim_df = simulate_from_bundle(bundle)
    except Exception:
        sim_df = None

    return jogos, pd.DataFrame(log_rows), preds, sim_df
