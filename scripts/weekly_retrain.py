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
    SHEET_COEFS_REG,
    SHEET_CONTEXTO,
    SHEET_FORECASTS,
    SHEET_LEIA_ME,
    SHEET_METRICAS,
    SHEET_OVERLAY,
    SHEET_PROJ_PROB,
    SHEET_PROJ_REG,
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

    bundle = train_pipeline(
        matches, cfg, run_backtest=not args.no_backtest, progress=_prog
    )
    try:
        save_offline_artifacts(
            bundle, forecasts=[], standings=None, calendar_proj=None, cfg=cfg
        )
        logger.info("Bundle intermediário salvo")
    except Exception as e:
        logger.warning("Bundle intermediário: %s", e)

    # --- Probabilístico: previsões + MC ---
    forecasts = forecast_calendar_games(bundle, jogos)
    logger.info("Prob: %s jogos pendentes", len(forecasts))
    cal_rows = []
    for p in forecasts:
        try:
            dh, ih = context_for_team(str(p["home_team"]), p.get("date"))
            da, ia = context_for_team(str(p["away_team"]), p.get("date"))
        except Exception:
            dh, ih, da, ia = 7.0, "Não tem", 7.0, "Não tem"
        cal_rows.append(
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
    cal_prob = pd.DataFrame(cal_rows)
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
    standings = None
    if fixtures:
        res = simulate_season(
            teams, current, fixtures, n_sims=budget_n_sims(cfg), seed=42
        )
        standings = result_to_frame(res)

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
    r_ini, r_fim = 1, 38
    jogos_reg, log_reg = aplicar_projecoes(
        jogos, "regressao_completa", r_ini, r_fim, "mandante_visitante"
    )
    classif_reg = tabela_comparativa_posicoes(jogos, jogos_reg)
    coefs_reg = tabela_regressao_acumulada_resumo(
        jogos, r_ini, r_fim, "completa"
    )
    logger.info("Regressão: %s projeções calculadas", len(log_reg))

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
