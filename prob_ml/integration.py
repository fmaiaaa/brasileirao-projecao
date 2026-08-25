"""Ponte entre o pipeline probabilístico e a lista de Jogo do app."""
from __future__ import annotations

from typing import Any

import pandas as pd

from brasileirao_projecao_core import Jogo, times_do_calendario
from prob_ml.config import load_config
from prob_ml.data import matches_from_calendar_jogos
from prob_ml.pipeline import FittedBundle, predict_fixtures, simulate_from_bundle, train_pipeline


def fit_from_jogos(
    jogos: list[Jogo],
    *,
    run_backtest: bool = True,
    cfg: dict[str, Any] | None = None,
) -> FittedBundle:
    """Treina somente quando chamado (modo selecionado no app)."""
    cfg = cfg or load_config()
    matches = matches_from_calendar_jogos(jogos)
    # tenta base FPT local se existir
    from pathlib import Path

    fpt = Path(__file__).resolve().parent.parent / cfg.get("data", {}).get(
        "local_path", "dados/fpt_matches.parquet"
    )
    if fpt.exists():
        from prob_ml.data import LocalFileDataSource

        try:
            matches, _ = LocalFileDataSource(fpt).load_canonical()
        except Exception:
            pass
    return train_pipeline(matches, cfg, run_backtest=run_backtest)


def aplicar_projecoes_probabilisticas(
    jogos: list[Jogo],
    bundle: FittedBundle,
) -> tuple[list[Jogo], pd.DataFrame, list[dict], Any]:
    """
    Preenche proj_pm/proj_pv com expected points da distribuição de placar.
    Retorna jogos, log, previsões detalhadas, resultado MC (ou None).
    """
    jogos = [Jogo(**j.__dict__) for j in jogos]
    preds = predict_fixtures(bundle)
    # indexar por mandante/visitante/rodada
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
            # fallback por times apenas
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
