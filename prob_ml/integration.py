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
    """
    cfg = cfg or load_config()
    data_cfg = cfg.get("data", {})
    report: dict[str, Any] = {"ok": False}

    file_id = (data_cfg.get("google_drive_file_id") or "").strip()
    file_url = (data_cfg.get("google_drive_file_url") or "").strip()
    source = str(data_cfg.get("source", "local")).lower()
    local_path = _ROOT / data_cfg.get("local_path", "dados/fpt_matches.csv")

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
            return matches, report
        except Exception as e:
            logger.warning("Drive indisponível (%s); tentando local", e)
            report["drive_error"] = f"{type(e).__name__}: {e}"

    if local_path.exists():
        matches, report = LocalFileDataSource(local_path).load_canonical()
        report["ok"] = True
        report["fallback"] = "local_file"
        # tenta Drive em background informativo
        if file_id or file_url:
            report["drive_configured"] = True
            report["drive_file_id"] = file_id or None
        return matches, report

    # tentativa Drive mesmo com source=local (se local ausente)
    if file_id or file_url:
        try:
            ds = GoogleDriveDataSource(
                file_id=file_id or None,
                file_url=file_url or None,
                cache_dir=_ROOT / data_cfg.get("cache_dir", "artifacts/prob_ml/cache"),
            )
            matches, report = ds.load_canonical()
            report["ok"] = True
            return matches, report
        except Exception as e:
            report["drive_error"] = f"{type(e).__name__}: {e}"

    if jogos is not None:
        matches = matches_from_calendar_jogos(jogos)
        report = {
            "ok": True,
            "fallback": "calendar",
            "n_rows": len(matches),
            "warning": "Base FPT ausente; usando calendário do app",
        }
        return matches, report

    raise FileNotFoundError(
        "Nenhuma base disponível. Configure Drive (compartilhe com a SA) "
        "ou coloque dados/fpt_matches.csv"
    )


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
    return bundle


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
