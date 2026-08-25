#!/usr/bin/env python
"""
Backtest de ranking: a cada rodada, posição final estimada vs ranking final real.
Gera XLSX com uma aba por temporada + aba de métricas OOF.

Uso:
  python scripts/backtest_ranking_xlsx.py
  python scripts/backtest_ranking_xlsx.py --n-sims 800 --out artifacts/prob_ml/backtest_ranking.xlsx
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prob_ml.config import load_config  # noqa: E402
from prob_ml.data import LocalFileDataSource  # noqa: E402
from prob_ml.ensemble import blend_distributions, optimize_blend_weights  # noqa: E402
from prob_ml.features import build_pre_match_features  # noqa: E402
from prob_ml.models import build_model_zoo  # noqa: E402
from prob_ml.ratings import update_elo  # noqa: E402
from prob_ml.simulation import simulate_season  # noqa: E402

logger = logging.getLogger("backtest_ranking")


def _standings_from_results(df: pd.DataFrame) -> pd.DataFrame:
    """Tabela a partir de jogos com placar (home_goals/away_goals)."""
    teams = sorted(
        set(df["home_team"].astype(str)) | set(df["away_team"].astype(str))
    )
    rows = {
        t: {"Time": t, "Pts": 0, "Vit": 0, "SG": 0, "GP": 0, "JC": 0} for t in teams
    }
    for _, r in df.iterrows():
        if pd.isna(r["home_goals"]) or pd.isna(r["away_goals"]):
            continue
        h, a = str(r["home_team"]), str(r["away_team"])
        hg, ag = int(r["home_goals"]), int(r["away_goals"])
        rows[h]["JC"] += 1
        rows[a]["JC"] += 1
        rows[h]["GP"] += hg
        rows[a]["GP"] += ag
        rows[h]["SG"] += hg - ag
        rows[a]["SG"] += ag - hg
        if hg > ag:
            rows[h]["Pts"] += 3
            rows[h]["Vit"] += 1
        elif hg < ag:
            rows[a]["Pts"] += 3
            rows[a]["Vit"] += 1
        else:
            rows[h]["Pts"] += 1
            rows[a]["Pts"] += 1
    tab = pd.DataFrame(list(rows.values()))
    tab = tab.sort_values(
        ["Pts", "Vit", "SG", "GP", "Time"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    tab.insert(0, "Ranking Final", np.arange(1, len(tab) + 1))
    return tab


def _current_dict(played: pd.DataFrame, teams: list[str]) -> dict[str, dict[str, float]]:
    cur = {t: {"points": 0.0, "wins": 0.0, "gd": 0.0, "gf": 0.0} for t in teams}
    for _, r in played.iterrows():
        if pd.isna(r["home_goals"]) or pd.isna(r["away_goals"]):
            continue
        h, a = str(r["home_team"]), str(r["away_team"])
        hg, ag = float(r["home_goals"]), float(r["away_goals"])
        if h not in cur or a not in cur:
            continue
        if hg > ag:
            cur[h]["points"] += 3
            cur[h]["wins"] += 1
        elif hg < ag:
            cur[a]["points"] += 3
            cur[a]["wins"] += 1
        else:
            cur[h]["points"] += 1
            cur[a]["points"] += 1
        cur[h]["gd"] += hg - ag
        cur[a]["gd"] += ag - hg
        cur[h]["gf"] += hg
        cur[a]["gf"] += ag
    return cur


def _fit_fold(
    matches: pd.DataFrame,
    feats: pd.DataFrame,
    train_idx: np.ndarray,
    *,
    max_goals: int,
    enabled: list[str] | None,
) -> tuple[list, np.ndarray]:
    models = build_model_zoo(max_goals, enabled)
    tr = matches.loc[train_idx]
    fr = feats.loc[train_idx]
    for m in models:
        m.fit(tr, fr)
    # pesos OOF leves no próprio treino (últimos 30%)
    idx = tr.dropna(subset=["home_goals", "away_goals"]).index.to_numpy()
    if len(idx) < 40:
        w = np.ones(len(models)) / len(models)
        return models, w
    cut = int(len(idx) * 0.7)
    te = idx[cut:]
    oof = [[] for _ in models]
    hg, ag = [], []
    for j in te:
        row_d = []
        for mi, m in enumerate(models):
            d = m.predict_match(matches.loc[j], feats.loc[j])
            oof[mi].append(d)
            row_d.append(d)
        hg.append(int(matches.loc[j, "home_goals"]))
        ag.append(int(matches.loc[j, "away_goals"]))
    w = optimize_blend_weights(oof, np.array(hg), np.array(ag), method="performance_weighted")
    return models, w


def _predict_indices(
    models: list,
    weights: np.ndarray,
    matches: pd.DataFrame,
    feats: pd.DataFrame,
    indices: np.ndarray,
) -> dict[int, object]:
    out: dict[int, object] = {}
    for j in indices:
        dists = [m.predict_match(matches.loc[j], feats.loc[j]) for m in models]
        out[int(j)] = blend_distributions(dists, weights)
    return out


def season_ranking_sheet(
    matches: pd.DataFrame,
    feats: pd.DataFrame,
    season: int,
    *,
    max_goals: int,
    enabled: list[str] | None,
    n_sims: int,
    seed: int,
    from_round: int = 19,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Ranking estimado vs final a partir de ``from_round`` (default 19 = 2º turno).

    Sem leakage:
    - treino = temporadas anteriores; se não houver, só rodadas < from_round da própria temporada
    - em cada rodada r >= from_round: tabela real até r + MC dos jogos restantes
    - features/previsões usam índices pré-jogo (pipeline leakage-safe)
    """
    seas = matches[matches["season"] == season].copy()
    seas = seas.dropna(subset=["home_goals", "away_goals"])
    if seas.empty or seas["round"].isna().all():
        raise ValueError(f"Temporada {season} sem jogos/rodadas válidos")

    max_r = int(seas["round"].max())
    if max_r < from_round:
        raise ValueError(
            f"Temporada {season} só vai até R{max_r}; from_round={from_round}"
        )

    teams = sorted(
        set(seas["home_team"].astype(str)) | set(seas["away_team"].astype(str))
    )
    final_tab = _standings_from_results(seas)
    final_rank = dict(zip(final_tab["Time"], final_tab["Ranking Final"]))

    # Treino sem ver o 2º turno da temporada-alvo
    prior = matches.index[
        (matches["season"] < season)
        & matches["home_goals"].notna()
        & matches["away_goals"].notna()
    ].to_numpy()
    note = "train_prior_seasons"
    if len(prior) < 50:
        boot = seas.index[seas["round"] < from_round].to_numpy()
        prior = boot
        note = f"train_rounds_1_to_{from_round - 1}_same_season"

    logger.info(
        "Season %s: train=%s (%s), eval R%s–R%s",
        season,
        len(prior),
        note,
        from_round,
        max_r,
    )
    models, weights = _fit_fold(
        matches, feats, prior, max_goals=max_goals, enabled=enabled
    )

    # Só prevê jogos da temporada (features já são pré-jogo)
    pred_idx = seas.index.to_numpy()
    dists = _predict_indices(models, weights, matches, feats, pred_idx)
    logger.info("Season %s: %s previsões", season, len(dists))

    pos_by_round: dict[int, dict[str, float]] = {}
    mae_list: list[float] = []

    for r in range(from_round, max_r + 1):
        played = seas[seas["round"] <= r]
        remaining = seas[seas["round"] > r]
        current = _current_dict(played, teams)
        fixtures = []
        for j in remaining.index:
            row = matches.loc[j]
            dist = dists.get(int(j))
            if dist is None:
                continue
            fixtures.append((str(row["home_team"]), str(row["away_team"]), dist))

        if not fixtures:
            est = {t: float(final_rank[t]) for t in teams}
        else:
            res = simulate_season(
                teams, current, fixtures, n_sims=n_sims, seed=seed + r
            )
            est = {t: float(p) for t, p in zip(res.teams, res.position_mean)}

        pos_by_round[r] = est
        err = [abs(est[t] - final_rank[t]) for t in teams]
        mae_list.append(float(np.mean(err)))

    rows = []
    for _, fr in final_tab.iterrows():
        t = fr["Time"]
        row = {
            "Ranking Final": int(fr["Ranking Final"]),
            "Time": t,
            "Pts Finais": int(fr["Pts"]),
        }
        for r in range(from_round, max_r + 1):
            row[f"Rodada {r}"] = round(pos_by_round[r][t], 2)
        rows.append(row)

    sheet = pd.DataFrame(rows)
    if f"Rodada {from_round}" in sheet.columns:
        sheet["Erro Abs R19"] = (
            sheet[f"Rodada {from_round}"].astype(float)
            - sheet["Ranking Final"].astype(float)
        ).abs().round(2)
        sheet["Proj Final (c/ dados até R19)"] = sheet[f"Rodada {from_round}"].round(2)
        sheet["Real Final (R38)"] = sheet["Ranking Final"]
    summary = {
        "season": float(season),
        "n_teams": float(len(teams)),
        "from_round": float(from_round),
        "max_round": float(max_r),
        "mae_pos_media": float(np.mean(mae_list)) if mae_list else float("nan"),
        "mae_pos_r19": float(
            np.mean([abs(pos_by_round[19][t] - final_rank[t]) for t in teams])
        )
        if 19 in pos_by_round
        else float("nan"),
        "mae_pos_r30": float(
            np.mean([abs(pos_by_round[30][t] - final_rank[t]) for t in teams])
        )
        if 30 in pos_by_round
        else float("nan"),
        "mae_pos_r38": float(
            np.mean([abs(pos_by_round[max_r][t] - final_rank[t]) for t in teams])
        )
        if max_r in pos_by_round
        else float("nan"),
        "train_mode": 0.0 if note.startswith("train_prior") else 1.0,
        "n_sims": float(n_sims),
    }
    return sheet, summary


def load_oof_metrics(cfg: dict) -> pd.DataFrame:
    path = ROOT / "artifacts" / "prob_ml" / "status.json"
    if not path.exists():
        return pd.DataFrame({"info": ["status.json ausente"]})
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    models = (data.get("metrics") or {}).get("models") or {}
    for name, met in models.items():
        rows.append({"Modelo": name, **met})
    ens = (data.get("metrics") or {}).get("ensemble")
    if ens:
        rows.append({"Modelo": "ensemble (champion)", **ens})
    df = pd.DataFrame(rows)
    meta = pd.DataFrame(
        [
            {
                "Modelo": "_meta",
                "score_nll": np.nan,
                "ll_1x2": np.nan,
                "brier": np.nan,
                "rps": np.nan,
                "n": data.get("metrics", {}).get("ensemble", {}).get("n"),
                "champion": data.get("champion"),
                "status": data.get("status"),
                "runtime_sec": data.get("runtime_sec"),
                "nota": "Menor score_nll / brier / rps = melhor; ll_1x2 menor (NLL) = melhor",
            }
        ]
    )
    # padroniza colunas
    for c in ("score_nll", "ll_1x2", "brier", "rps", "n"):
        if c not in df.columns:
            df[c] = np.nan
    return pd.concat([df, meta], ignore_index=True)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "prob_ml" / "backtest_ranking.xlsx",
    )
    p.add_argument("--n-sims", type=int, default=1200)
    p.add_argument("--seasons", type=str, default="")
    p.add_argument(
        "--from-round",
        type=int,
        default=19,
        help="Primeira rodada reportada (default 19 = 2º turno, anti-leakage)",
    )
    p.add_argument(
        "--serie-a-only",
        action="store_true",
        default=True,
        help="Só Série A no ranking por temporada (default True)",
    )
    args = p.parse_args()

    cfg = load_config()
    local = ROOT / cfg.get("data", {}).get("local_path", "dados/fpt_matches.csv")
    matches, report = LocalFileDataSource(local).load_canonical()
    logger.info("rows=%s fingerprint=%s", len(matches), report.get("fingerprint"))

    # Ranking é do Brasileirão Série A
    if args.serie_a_only and "competition" in matches.columns:
        from brasileirao_multi_liga import normalize_competition

        comps = matches["competition"].map(normalize_competition)
        matches = matches.loc[comps.eq("serie_a")].copy()
        logger.info("Filtro serie_a → %s linhas", len(matches))

    matches, _ = update_elo(matches)
    feats, _ = build_pre_match_features(
        matches,
        rolling_windows=cfg.get("features", {}).get("rolling_windows", [3, 5, 8]),
        ewma_halflives=cfg.get("features", {}).get("ewma_halflives", [3, 5, 10]),
    )

    seasons = sorted(matches["season"].dropna().unique())
    if args.seasons.strip():
        want = {int(x) for x in args.seasons.split(",")}
        seasons = [s for s in seasons if int(s) in want]

    max_goals = int(cfg.get("goal_support", 8))
    enabled = cfg.get("models_enabled")
    summaries = []
    sheets: dict[str, pd.DataFrame] = {}

    for s in seasons:
        s_int = int(s)
        try:
            sheet, summary = season_ranking_sheet(
                matches,
                feats,
                s_int,
                max_goals=max_goals,
                enabled=enabled,
                n_sims=args.n_sims,
                seed=42 + s_int,
                from_round=args.from_round,
            )
            sheets[str(s_int)] = sheet
            summaries.append(summary)
            logger.info(
                "Season %s MAE média pos=%.2f",
                s_int,
                summary["mae_pos_media"],
            )
        except Exception as e:
            logger.exception("Falha temporada %s: %s", s_int, e)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    metricas = load_oof_metrics(cfg)
    rank_sum = pd.DataFrame(summaries)
    if not rank_sum.empty:
        rank_sum["train_mode"] = rank_sum["train_mode"].map(
            {0.0: "temporadas_anteriores", 1.0: f"bootstrap_R1_a_R{args.from_round - 1}"}
        )

    with pd.ExcelWriter(args.out, engine="openpyxl") as xw:
        pd.DataFrame(
            [
                {
                    "nota": (
                        f"Colunas R{args.from_round:02d}+ apenas. "
                        "Treino sem ver o 2º turno da temporada-alvo "
                        "(temporadas anteriores ou R1..R18). Sem data leakage."
                    )
                }
            ]
        ).to_excel(xw, sheet_name="Leia-me", index=False)
        metricas.to_excel(xw, sheet_name="Metricas_OOF", index=False)
        if not rank_sum.empty:
            rank_sum.to_excel(xw, sheet_name="Resumo_Ranking", index=False)
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name, index=False)

    dl = Path.home() / "Downloads" / "backtest_ranking_R19plus_brasileirao.xlsx"
    try:
        import shutil

        shutil.copy2(args.out, dl)
        logger.info("Cópia: %s", dl)
    except Exception as e:
        logger.warning("Não copiou para Downloads: %s", e)

    logger.info("XLSX: %s", args.out)
    print(f"OK {args.out}")
    if not rank_sum.empty:
        print(rank_sum.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
