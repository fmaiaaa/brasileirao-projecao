#!/usr/bin/env python
"""
Retreino semanal (segundas):
  1) Baixa FPT multi-liga
  2) Carrega calendário (Sheets) e faz overlay de placares frescos na FPT
  3) Treina probabilístico + gera projeções da regressão
  4) Salva brasileirao_modelos.xlsx (dados/ + Downloads) — app só lê este XLSX
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

load_dotenv(ROOT / ".env")
load_dotenv(ROOT / ".env", override=True)
load_dotenv(Path.home() / "Downloads" / "football-ml-predictor" / ".env")

from brasileirao_projecao_core import (  # noqa: E402
    aplicar_projecoes,
    carregar_jogos,
    classificacao,
    stats_acumuladas_ate,
    tabela_comparativa_posicoes,
    tabela_regressao_acumulada_resumo,
    times_do_calendario,
)
from entrega_xlsx import (  # noqa: E402
    MODELOS_XLSX_NAME,
    SHEET_CLASSIF_PROB,
    SHEET_CLASSIF_REG,
    SHEET_CLASSIF_MODELOS,
    SHEET_COEFS_MODELOS,
    SHEET_COEFS_REG,
    SHEET_CONTEXTO,
    SHEET_FORECASTS,
    SHEET_FORECASTS_MODELOS,
    SHEET_LEIA_ME,
    SHEET_METRICAS,
    SHEET_OVERLAY,
    SHEET_PROJ_MODELOS,
    SHEET_PROJ_PROB,
    SHEET_PROJ_REG,
    SHEET_RESUMO_MODELOS,
    build_entrega_xlsx,
)
from brasileirao_gsheets import (  # noqa: E402
    load_service_account_info,
    publicar_modelos_na_planilha,
    upload_xlsx_drive,
)
from prob_ml.calendar_overlay import overlay_fpt_with_calendar  # noqa: E402
from prob_ml.config import budget_n_sims, load_config  # noqa: E402
from prob_ml.context_calendar import (  # noqa: E402
    attach_context_to_matches,
    clear_context_cache,
    context_for_team,
    download_fpt_cups,
)
from prob_ml.data import LocalFileDataSource  # noqa: E402
from prob_ml.fpt_leagues import download_fpt_multi, leagues_from_config  # noqa: E402
from prob_ml.offline import forecast_calendar_games, save_offline_artifacts  # noqa: E402
from prob_ml.pipeline import train_pipeline  # noqa: E402
from prob_ml.simulation import result_to_frame, simulate_season  # noqa: E402
from brasileirao_projecao_core import Jogo  # noqa: E402

logger = logging.getLogger("weekly_retrain")

MODELOS_ACUM_JANELAS = ("2025", "ultimas_38_rodadas", "ultimos_3_anos")
MODELOS_ACUM_MODOS = (
    ("Regressão FE", "regressao_completa"),
    ("Kalman", "kalman_acumulada"),
    ("XGBoost", "xgboost_acumulada"),
    ("GAM", "gam_acumulada"),
)


def _gerar_modelos_acumulados(
    jogos: list[Jogo],
    r_ini: int,
    r_fim: int,
    *,
    ano_calendario: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Projeções, resumo e coeficientes para FE/Kalman/XGBoost/GAM × 3 janelas."""
    from recency import JANELA_TREINO_LABELS, JanelaTreino
    from modelos_acumulados import (
        ajustar_modelo_acumulado,
        coeficientes_para_base,
        coletar_painel_treino,
        exportar_coefs_regressao_fe,
        tabela_resumo_modelo,
    )

    proj_rows: list[dict] = []
    resumo_rows: list[dict] = []
    coef_parts: list[pd.DataFrame] = []

    for janela in MODELOS_ACUM_JANELAS:
        janela_t: JanelaTreino = janela  # type: ignore[assignment]
        janela_lbl = JANELA_TREINO_LABELS[janela_t]
        for nome, modo in MODELOS_ACUM_MODOS:
            logger.info("Modelo acumulado: %s | janela=%s", nome, janela_lbl)
            try:
                _, log_df = aplicar_projecoes(
                    jogos,
                    modo,  # type: ignore[arg-type]
                    r_ini,
                    r_fim,
                    "mandante_visitante",
                    janela=janela_t,
                    ano_calendario=ano_calendario,
                )
                for _, row in log_df.iterrows():
                    proj_rows.append(
                        {
                            "Modelo": nome,
                            "Janela": janela_lbl,
                            "Rodada": row.get("Rodada"),
                            "Mandante": row.get("Mandante"),
                            "Visitante": row.get("Visitante"),
                            "Proj": row.get("Proj"),
                        }
                    )
                if modo == "regressao_completa":
                    coef_df = exportar_coefs_regressao_fe(
                        jogos,
                        r_ini,
                        r_fim,
                        janela_t,
                        ano_calendario=ano_calendario,
                        janela_lbl=janela_lbl,
                    )
                    if coef_df is not None and not coef_df.empty:
                        coef_parts.append(coef_df)
                    coefs = tabela_regressao_acumulada_resumo(
                        jogos,
                        r_ini,
                        r_fim,
                        "completa",
                        janela=janela_t,
                        ano_calendario=ano_calendario,
                    )
                    r2_val = None
                    if coefs is not None and not coefs.empty and "R²" in coefs.columns:
                        r2_val = coefs["R²"].iloc[0]
                    resumo_rows.append(
                        {
                            "Modelo": nome,
                            "Janela": janela_lbl,
                            "N observações": coefs.shape[0] if coefs is not None else 0,
                            "R²": r2_val,
                        }
                    )
                else:
                    tipo = modo.replace("_acumulada", "")
                    painel = coletar_painel_treino(
                        jogos, janela_t, ano_calendario=ano_calendario
                    )
                    modelo = ajustar_modelo_acumulado(painel, tipo, janela_t)  # type: ignore[arg-type]
                    cbase = coeficientes_para_base(modelo, nome, janela_lbl)
                    if cbase is not None and not cbase.empty:
                        coef_parts.append(cbase)
                    resumo = tabela_resumo_modelo(modelo)
                    n_obs = resumo.loc[resumo["Campo"] == "N observações", "Valor"]
                    r2 = resumo.loc[resumo["Campo"] == "R²", "Valor"]
                    resumo_rows.append(
                        {
                            "Modelo": nome,
                            "Janela": janela_lbl,
                            "N observações": n_obs.iloc[0] if len(n_obs) else None,
                            "R²": r2.iloc[0] if len(r2) else None,
                        }
                    )
            except Exception as e:
                logger.warning("Falha %s / %s: %s", nome, janela_lbl, e)
                resumo_rows.append(
                    {
                        "Modelo": nome,
                        "Janela": janela_lbl,
                        "N observações": None,
                        "R²": None,
                        "Erro": str(e)[:200],
                    }
                )

    coefs_all = (
        pd.concat(coef_parts, ignore_index=True) if coef_parts else pd.DataFrame()
    )
    return pd.DataFrame(proj_rows), pd.DataFrame(resumo_rows), coefs_all


def _matches_para_janela(
    matches: pd.DataFrame,
    janela: str,
    cal_teams: list[str],
    ano_cal: int,
    cfg,
) -> pd.DataFrame:
    from recency import (
        JanelaTreino,
        attach_sample_weights,
        filter_matches_by_janela,
        load_recency_settings,
    )

    janela_t: JanelaTreino = janela  # type: ignore[assignment]
    rcfg = load_recency_settings(cfg)
    m = filter_matches_by_janela(
        matches,
        janela_t,
        ano_calendario=int(ano_cal),
        calendar_teams=cal_teams,
        history_rounds=int(rcfg["history_rounds"]),
    )
    return attach_sample_weights(
        m,
        calendar_teams=cal_teams,
        current_season=int(ano_cal),
        janela=janela_t,
        ano_calendario=int(ano_cal),
    )


def _gerar_media_prob_janelas(
    matches: pd.DataFrame,
    jogos: list[Jogo],
    cfg,
    *,
    ano_calendario: int,
    r_ini: int,
    r_fim: int,
    run_backtest: bool,
    progress,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    object,
    list,
    pd.DataFrame,
    pd.DataFrame | None,
]:
    """Média + Probabilístico × 3 janelas → projeções, resumo, coefs, classif, forecasts."""
    import json

    from recency import JANELA_TREINO_LABELS, JanelaTreino
    from brasileirao_projecao_core import (
        aplicar_projecoes,
        export_medias_para_base,
        jogos_treino_por_janela,
        stats_acumuladas_ate,
    )
    from brasileirao_sheet_names import LABEL_MEDIA, LABEL_PROB
    from prob_ml.config import budget_n_sims
    from prob_ml.context_calendar import context_for_team
    from prob_ml.offline import forecast_calendar_games
    from prob_ml.pipeline import train_pipeline
    from prob_ml.simulation import result_to_frame, simulate_season

    cal_teams = times_do_calendario(jogos)
    proj_parts: list[pd.DataFrame] = []
    resumo_parts: list[dict] = []
    coef_parts: list[pd.DataFrame] = []
    classif_parts: list[pd.DataFrame] = []
    fc_parts: list[pd.DataFrame] = []

    bundle_ref = None
    forecasts_ref: list = []
    cal_prob_ref = pd.DataFrame()
    standings_ref = None

    for janela in MODELOS_ACUM_JANELAS:
        janela_t: JanelaTreino = janela  # type: ignore[assignment]
        janela_lbl = JANELA_TREINO_LABELS[janela_t]
        logger.info("Média + Prob | janela=%s", janela_lbl)

        try:
            _, log_media = aplicar_projecoes(
                jogos,
                "media_simples",
                r_ini,
                r_fim,
                "mandante_visitante",
                janela=janela_t,
                ano_calendario=ano_calendario,
            )
            if not log_media.empty:
                media_df = log_media.copy()
                media_df.insert(0, "Janela", janela_lbl)
                media_df.insert(0, "Modelo", LABEL_MEDIA)
                proj_parts.append(media_df[["Modelo", "Janela", "Rodada", "Mandante", "Visitante", "Proj"]])
            n_media = len(
                jogos_treino_por_janela(jogos, janela_t, ano_calendario=ano_calendario)
            )
            resumo_parts.append(
                {
                    "Modelo": LABEL_MEDIA,
                    "Janela": janela_lbl,
                    "N observações": n_media,
                    "R²": None,
                }
            )
            coef_m = export_medias_para_base(
                jogos, r_ini, r_fim, janela_t, janela_lbl, ano_calendario=ano_calendario
            )
            if coef_m is not None and not coef_m.empty:
                coef_parts.append(coef_m)
        except Exception as e:
            logger.warning("Média / %s: %s", janela_lbl, e)

        try:
            matches_j = _matches_para_janela(
                matches, janela, cal_teams, ano_calendario, cfg
            )
            do_bt = run_backtest and janela_t == "ultimas_38_rodadas"
            bundle_j = train_pipeline(
                matches_j, cfg, run_backtest=do_bt, progress=progress
            )
            forecasts_j = forecast_calendar_games(bundle_j, jogos)
            prob_rows: list[dict] = []
            legacy_cal: list[dict] = []
            for p in forecasts_j:
                try:
                    dh, ih = context_for_team(str(p["home_team"]), p.get("date"))
                    da, ia = context_for_team(str(p["away_team"]), p.get("date"))
                except Exception:
                    dh, ih, da, ia = 7.0, "Não tem", 7.0, "Não tem"
                prob_rows.append(
                    {
                        "Modelo": LABEL_PROB,
                        "Janela": janela_lbl,
                        "Rodada": p["round"],
                        "Mandante": p["home_team"],
                        "Visitante": p["away_team"],
                        "Proj": f"{p['xpts_home']:.2f} / {p['xpts_away']:.2f}",
                    }
                )
                if janela_t == "ultimas_38_rodadas":
                    legacy_cal.append(
                        {
                            "Rodada": p["round"],
                            "Mandante": p["home_team"],
                            "Visitante": p["away_team"],
                            "Proj": f"{p['xpts_home']:.2f} / {p['xpts_away']:.2f}",
                            "λ": f"{p['xg_home']:.2f} x {p['xg_away']:.2f}",
                            "1X2": f"{p['p_home']:.0%} / {p['p_draw']:.0%} / {p['p_away']:.0%}",
                            "xpts_home": p["xpts_home"],
                            "xpts_away": p["xpts_away"],
                            "Descanso M": dh,
                            "Descanso V": da,
                            "Importante M": ih,
                            "Importante V": ia,
                        }
                    )
            if prob_rows:
                proj_parts.append(pd.DataFrame(prob_rows))

            teams = cal_teams
            current = {
                t: {"points": 0.0, "wins": 0.0, "gd": 0.0, "gf": 0.0} for t in teams
            }
            for t in teams:
                st = stats_acumuladas_ate(jogos, t, 38, so_realizados=True)
                current[t] = {
                    "points": float(st.pts),
                    "wins": float(st.vit),
                    "gd": float(st.sg),
                    "gf": float(st.gf),
                }
            fixtures = [
                (p["home_team"], p["away_team"], p["dist"])
                for p in forecasts_j
                if "dist" in p
            ]
            standings_j = None
            if fixtures:
                res = simulate_season(
                    teams, current, fixtures, n_sims=budget_n_sims(cfg), seed=42
                )
                standings_j = result_to_frame(res)
                st_df = standings_j.copy()
                st_df.insert(0, "Janela", janela_lbl)
                st_df.insert(0, "Modelo", LABEL_PROB)
                classif_parts.append(st_df)

            metrics = bundle_j.status.metrics or {}
            ens_r2 = None
            if isinstance(metrics, dict):
                ens = metrics.get("ensemble") or {}
                ens_r2 = ens.get("r2") or ens.get("R2")
            resumo_parts.append(
                {
                    "Modelo": LABEL_PROB,
                    "Janela": janela_lbl,
                    "N observações": len(matches_j),
                    "R²": ens_r2,
                }
            )

            fc_rows = []
            for p in forecasts_j:
                fc_rows.append(
                    {
                        "Modelo": LABEL_PROB,
                        "Janela": janela_lbl,
                        "round": p.get("round"),
                        "date": p.get("date"),
                        "home_team": p["home_team"],
                        "away_team": p["away_team"],
                        "xg_home": p["xg_home"],
                        "xg_away": p["xg_away"],
                        "p_home": p["p_home"],
                        "p_draw": p["p_draw"],
                        "p_away": p["p_away"],
                        "xpts_home": p["xpts_home"],
                        "xpts_away": p["xpts_away"],
                        "over_25": p.get("over_25"),
                        "btts_yes": p.get("btts_yes"),
                        "top_scores": json.dumps(p.get("top_scores") or []),
                    }
                )
            if fc_rows:
                fc_parts.append(pd.DataFrame(fc_rows))

            if janela_t == "ultimas_38_rodadas":
                bundle_ref = bundle_j
                forecasts_ref = forecasts_j
                cal_prob_ref = pd.DataFrame(legacy_cal)
                standings_ref = standings_j
        except Exception as e:
            logger.warning("Prob / %s: %s", janela_lbl, e)
            resumo_parts.append(
                {
                    "Modelo": LABEL_PROB,
                    "Janela": janela_lbl,
                    "N observações": None,
                    "R²": None,
                    "Erro": str(e)[:200],
                }
            )

    proj_all = pd.concat(proj_parts, ignore_index=True) if proj_parts else pd.DataFrame()
    resumo_all = pd.DataFrame(resumo_parts)
    coef_all = pd.concat(coef_parts, ignore_index=True) if coef_parts else pd.DataFrame()
    classif_all = (
        pd.concat(classif_parts, ignore_index=True) if classif_parts else pd.DataFrame()
    )
    fc_all = pd.concat(fc_parts, ignore_index=True) if fc_parts else pd.DataFrame()
    return (
        proj_all,
        resumo_all,
        coef_all,
        classif_all,
        fc_all,
        bundle_ref,
        forecasts_ref,
        cal_prob_ref,
        standings_ref,
    )


def _seasons_for_download(cfg) -> list[int]:
    from datetime import date
    from recency import allowed_seasons

    return sorted(allowed_seasons(ref=date.today()))


def _load_calendar() -> tuple[list[Jogo], str]:
    try:
        jogos, fonte, _ = carregar_jogos(preferir_gsheets=True)
        return jogos, fonte
    except Exception:
        jogos, fonte, _ = carregar_jogos(preferir_gsheets=False)
        return jogos, fonte


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    p = argparse.ArgumentParser(description="Retreino semanal Regressão + Probabilístico")
    p.add_argument("--budget", choices=["fast", "standard", "full"], default="fast")
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--no-backtest", action="store_true")
    args = p.parse_args()

    cfg = load_config()
    cfg["compute_budget"] = args.budget
    local_csv = ROOT / cfg.get("data", {}).get("local_path", "dados/fpt_matches.csv")

    if not args.skip_download:
        api_key = (os.environ.get("FPT_API_KEY") or "").strip()
        if not api_key:
            logger.error("FPT_API_KEY ausente (.env). Abortando download.")
            return 2
        seasons = _seasons_for_download(cfg)
        download_fpt_multi(
            api_key,
            local_csv,
            leagues=leagues_from_config(cfg),
            seasons=seasons,
        )
        try:
            download_fpt_cups(api_key)
            logger.info("Copas (Libertadores/Sul-Americana) atualizadas")
        except Exception as e:
            logger.warning("Copas FPT: %s", e)
        clear_context_cache()
    elif not local_csv.exists():
        logger.error("Arquivo local ausente: %s", local_csv)
        return 1
    else:
        api_key = (os.environ.get("FPT_API_KEY") or "").strip()
        if api_key:
            try:
                download_fpt_cups(api_key)
                clear_context_cache()
            except Exception as e:
                logger.warning("Copas FPT (skip-download): %s", e)

    matches, report = LocalFileDataSource(local_csv).load_canonical()
    jogos, fonte = _load_calendar()
    logger.info("Calendário: %s (%s jogos)", fonte, len(jogos))

    matches, overlay = overlay_fpt_with_calendar(matches, jogos)
    # persiste FPT já com overlay (útil se mid-week só rodar app)
    matches.to_csv(local_csv, index=False, encoding="utf-8-sig")
    clear_context_cache()
    logger.info(
        "FPT+overlay rows=%s updated=%s inserted=%s",
        len(matches),
        overlay.get("n_updated"),
        overlay.get("n_inserted"),
    )

    def _prog(frac: float, msg: str) -> None:
        logger.info("[%.0f%%] %s", 100 * frac, msg)

    r_ini, r_fim = 1, 38
    ano_cal = max(
        (
            int(str(j.data)[:4])
            for j in jogos
            if j.data and str(j.data)[:4].isdigit()
        ),
        default=2026,
    )

    (
        proj_mp,
        resumo_mp,
        coef_mp,
        classif_modelos,
        forecasts_modelos,
        bundle,
        forecasts,
        cal_prob,
        standings,
    ) = _gerar_media_prob_janelas(
        matches,
        jogos,
        cfg,
        ano_calendario=ano_cal,
        r_ini=r_ini,
        r_fim=r_fim,
        run_backtest=not args.no_backtest,
        progress=_prog,
    )
    if bundle is None:
        logger.error("Falha no treino probabilístico (38 rodadas).")
        return 1

    try:
        save_offline_artifacts(
            bundle, forecasts=[], standings=None, calendar_proj=None, cfg=cfg
        )
        logger.info("Bundle intermediário salvo")
    except Exception as e:
        logger.warning("Bundle intermediário: %s", e)

    logger.info("Prob: %s jogos pendentes", len(forecasts))
    if cal_prob.empty and forecasts:
        cal_rows = []
        for p in forecasts:
            cal_rows.append(
                {
                    "Rodada": p["round"],
                    "Mandante": p["home_team"],
                    "Visitante": p["away_team"],
                    "Proj": f"{p['xpts_home']:.2f} / {p['xpts_away']:.2f}",
                }
            )
        cal_prob = pd.DataFrame(cal_rows)

    forecasts_save = [{k: v for k, v in p.items() if k != "dist"} for p in forecasts]
    save_offline_artifacts(
        bundle,
        forecasts=forecasts_save,
        standings=standings,
        calendar_proj=cal_prob,
        cfg=cfg,
        extra_meta={
            "budget": args.budget,
            "download_skipped": args.skip_download,
            "calendar_source": fonte,
            "n_calendar_pending": len(forecasts),
            "calendar_overlay": overlay,
            "data_report": {
                k: report.get(k)
                for k in ("fingerprint", "n_rows", "source", "fallback", "ok")
            },
        },
    )

    # --- Regressão (efeitos fixos + multi-liga + contexto) ---
    jogos_reg, log_reg = aplicar_projecoes(
        jogos,
        "regressao_completa",
        r_ini,
        r_fim,
        "mandante_visitante",
        janela="ultimas_38_rodadas",
        ano_calendario=ano_cal,
    )
    classif_reg = tabela_comparativa_posicoes(jogos, jogos_reg)
    coefs_reg = tabela_regressao_acumulada_resumo(
        jogos,
        r_ini,
        r_fim,
        "completa",
        janela="ultimas_38_rodadas",
        ano_calendario=ano_cal,
    )
    logger.info("Regressão (38 rod.): %s projeções calculadas", len(log_reg))

    proj_modelos, resumo_modelos, coefs_modelos = _gerar_modelos_acumulados(
        jogos, r_ini, r_fim, ano_calendario=ano_cal
    )
    if not proj_mp.empty:
        proj_modelos = pd.concat([proj_modelos, proj_mp], ignore_index=True)
    if not resumo_mp.empty:
        resumo_modelos = pd.concat([resumo_modelos, resumo_mp], ignore_index=True)
    if not coef_mp.empty:
        coefs_modelos = pd.concat([coefs_modelos, coef_mp], ignore_index=True)
    logger.info(
        "Modelos (todos): %s projeções, %s resumos, %s coeficientes",
        len(proj_modelos),
        len(resumo_modelos),
        len(coefs_modelos),
    )

    classif_prob = standings if standings is not None else classificacao(jogos_reg)

    # Base completa com contexto (Série A + demais ligas FPT)
    try:
        base_contexto = attach_context_to_matches(matches)
        keep = [
            c
            for c in base_contexto.columns
            if c
            in {
                "date",
                "season",
                "competition",
                "round",
                "home_team",
                "away_team",
                "home_goals",
                "away_goals",
                "home_xg",
                "away_xg",
                "home_rest_days",
                "away_rest_days",
                "home_important",
                "away_important",
            }
            or c.startswith("home_imp_")
            or c.startswith("away_imp_")
        ]
        base_contexto = base_contexto[keep]
    except Exception as e:
        logger.warning("Base_Contexto: %s", e)
        base_contexto = None

    fc_df = None
    try:
        from prob_ml.offline import load_forecasts

        fc_df = load_forecasts()
    except Exception:
        fc_df = None
    if (fc_df is None or fc_df.empty) and forecasts:
        fc_df = pd.DataFrame(
            [
                {
                    "round": p.get("round"),
                    "date": p.get("date"),
                    "home_team": p["home_team"],
                    "away_team": p["away_team"],
                    "xg_home": p["xg_home"],
                    "xg_away": p["xg_away"],
                    "p_home": p["p_home"],
                    "p_draw": p["p_draw"],
                    "p_away": p["p_away"],
                    "xpts_home": p["xpts_home"],
                    "xpts_away": p["xpts_away"],
                    "over_25": p["over_25"],
                    "under_25": p["under_25"],
                    "btts_yes": p["btts_yes"],
                    "btts_no": p["btts_no"],
                    "top_scores": json.dumps(p.get("top_scores") or []),
                    "champion": p.get("champion"),
                    "status": p.get("status"),
                }
                for p in forecasts
            ]
        )

    stamp = datetime.now().strftime("%Y%m%d")
    entrega_dir = ROOT / "artifacts" / "entrega"
    dados_dir = ROOT / "dados"
    dados_dir.mkdir(parents=True, exist_ok=True)
    entrega_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "champion": bundle.status.champion,
        "status": bundle.status.status,
        "n_matches_train": len(matches),
        "calendar_source": fonte,
        "fingerprint": bundle.status.dataset_fingerprint,
        "runtime_sec": bundle.status.runtime_sec,
        "context_vars": "dias_descanso + jogos_importantes",
        "modelos_acumulados": "FE + Kalman + XGBoost + GAM + Média + Prob × 3 janelas",
    }
    # Destinos: dados/ (junto ao calendário) + entrega + Downloads
    destinos = [
        dados_dir / MODELOS_XLSX_NAME,
        entrega_dir / MODELOS_XLSX_NAME,
        entrega_dir / "brasileirao_modelos_latest.xlsx",
        entrega_dir / f"brasileirao_modelos_{stamp}.xlsx",
        Path.home() / "Downloads" / MODELOS_XLSX_NAME,
        Path.home() / "Downloads" / f"brasileirao_modelos_{stamp}.xlsx",
    ]
    primary = destinos[0]
    build_entrega_xlsx(
        primary,
        classif_reg=classif_reg,
        classif_prob=classif_prob,
        proj_reg=log_reg,
        proj_prob=cal_prob,
        metrics_prob=bundle.status.metrics,
        overlay_report=overlay,
        coefs_reg=coefs_reg,
        base_contexto=base_contexto,
        match_forecasts=fc_df,
        proj_modelos=proj_modelos,
        resumo_modelos=resumo_modelos,
        coefs_modelos=coefs_modelos,
        classif_modelos=classif_modelos,
        forecasts_modelos=forecasts_modelos,
        meta=meta,
    )
    for dest in destinos[1:]:
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(primary, dest)
        except Exception as e:
            logger.warning("Cópia %s: %s", dest, e)

    # --- Publica na planilha Google (mesma dos resultados) + Drive opcional ---
    try:
        info = load_service_account_info()
        if info is None:
            logger.warning(
                "Sem credenciais Google no agendador — "
                "defina GOOGLE_SERVICE_ACCOUNT_FILE no .env para publicação automática."
            )
        else:
            # métricas como DF (mesma lógica do XLSX)
            met_rows = []
            metrics_prob = bundle.status.metrics
            if metrics_prob and isinstance(metrics_prob, dict):
                if "models" in metrics_prob:
                    for name, m in (metrics_prob.get("models") or {}).items():
                        met_rows.append({"Modelo": name, **(m or {})})
                    ens = metrics_prob.get("ensemble")
                    if ens:
                        met_rows.append({"Modelo": "ensemble", **ens})
            met_df = pd.DataFrame(met_rows) if met_rows else pd.DataFrame({"info": ["sem métricas"]})
            leia_df = pd.DataFrame(
                [{"campo": k, "valor": str(v)} for k, v in {"gerado_em": datetime.now().isoformat(timespec="seconds"), **meta}.items()]
            )
            pub = publicar_modelos_na_planilha(
                {
                    SHEET_LEIA_ME: leia_df,
                    SHEET_PROJ_REG: log_reg,
                    SHEET_COEFS_REG: coefs_reg,
                    SHEET_CLASSIF_REG: classif_reg,
                    SHEET_PROJ_MODELOS: proj_modelos,
                    SHEET_RESUMO_MODELOS: resumo_modelos,
                    SHEET_COEFS_MODELOS: coefs_modelos,
                    SHEET_CLASSIF_MODELOS: classif_modelos,
                    SHEET_FORECASTS_MODELOS: forecasts_modelos,
                    SHEET_PROJ_PROB: cal_prob,
                    SHEET_FORECASTS: fc_df if fc_df is not None else pd.DataFrame(),
                    SHEET_CLASSIF_PROB: classif_prob,
                    SHEET_METRICAS: met_df,
                    SHEET_CONTEXTO: base_contexto if base_contexto is not None else pd.DataFrame(),
                    SHEET_OVERLAY: pd.DataFrame([overlay]),
                },
                service_account_info=info,
            )
            logger.info(
                "Sheets publicada: %s abas → %s",
                len(pub.get("sheets") or []),
                pub.get("spreadsheet_id"),
            )
            # Ranking ano a ano (R19+, sem leakage) — se artefato existir
            try:
                from publish_ranking_anos import load_year_sheets, CANDIDATES

                rk_path = next((p for p in CANDIDATES if p.exists()), None)
                if rk_path is not None:
                    year_sheets = load_year_sheets(rk_path)
                    pub2 = publicar_modelos_na_planilha(
                        year_sheets, service_account_info=info
                    )
                    logger.info("Ranking anos publicados: %s", pub2.get("sheets"))
                else:
                    logger.warning("Artefato ranking anos ausente — rode backtest_ranking_xlsx.py")
            except Exception as e:
                logger.warning("Publicação ranking anos: %s", e)
            drive_id = (os.environ.get("MODELOS_DRIVE_FILE_ID") or "").strip()
            if drive_id:
                up = upload_xlsx_drive(primary, file_id=drive_id, service_account_info=info)
                logger.info("Drive XLSX atualizado: %s", up.get("file_id"))
    except Exception as e:
        logger.warning("Publicação Google falhou: %s", e)

    # CSVs legados (opcional; app não usa)
    reg_dir = ROOT / "artifacts" / "regressao"
    reg_dir.mkdir(parents=True, exist_ok=True)
    log_reg.to_csv(reg_dir / "calendar_projecoes.csv", index=False, encoding="utf-8-sig")
    classif_reg.to_csv(reg_dir / "classificacao.csv", index=False, encoding="utf-8-sig")
    coefs_reg.to_csv(reg_dir / "coefs.csv", index=False, encoding="utf-8-sig")
    logger.info("Regressão: %s projeções · XLSX=%s", len(log_reg), primary)

    logger.info("ENTREGA XLSX: %s (+ cópias dados/Downloads)", primary)
    logger.info(
        "DONE status=%s champion=%s runtime=%ss",
        bundle.status.status,
        bundle.status.champion,
        bundle.status.runtime_sec,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
