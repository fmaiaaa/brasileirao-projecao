"""Pipelines de treino e previsão (sob demanda)."""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from prob_ml.backtesting import expanding_season_splits, nested_inner_splits
from prob_ml.backtesting.metrics import aggregate_metrics, score_nll
from prob_ml.calibration import Calibrator, fit_temperature, recalibrate_distribution
from prob_ml.config import artifacts_dir, budget_hpo_trials, budget_n_sims, load_config
from prob_ml.ensemble import blend_distributions, optimize_blend_weights
from prob_ml.features import build_pre_match_features
from prob_ml.models import build_model_zoo
from prob_ml.models.score_matrix import ScoreDistribution
from prob_ml.optimization import run_hpo
from prob_ml.ratings import update_elo
from prob_ml.selection import select_features
from prob_ml.simulation import SeasonSimResult, result_to_frame, simulate_season

logger = logging.getLogger(__name__)

ProgressCb = Callable[[float, str], None]


def _emit(progress: ProgressCb | None, frac: float, msg: str) -> None:
    if progress is None:
        return
    try:
        progress(float(min(max(frac, 0.0), 1.0)), msg)
    except Exception:
        pass


@dataclass
class ExperimentStatus:
    status: str = "not_evaluated"  # not_evaluated | trained | failed
    champion: str | None = None
    ensemble_method: str | None = None
    ensemble_weights: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    selected_features: list[str] = field(default_factory=list)
    dataset_fingerprint: str | None = None
    model_names: list[str] = field(default_factory=list)
    calibration: str = "none"
    notes: list[str] = field(default_factory=list)
    runtime_sec: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FittedBundle:
    models: list[Any]
    model_names: list[str]
    weights: np.ndarray
    calibrator: Calibrator
    features_frame: pd.DataFrame
    matches: pd.DataFrame
    status: ExperimentStatus
    selected_features: list[str]
    max_goals: int = 8


def _played_mask(df: pd.DataFrame) -> pd.Series:
    return df["home_goals"].notna() & df["away_goals"].notna()


def train_pipeline(
    matches: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
    *,
    run_backtest: bool = True,
    progress: ProgressCb | None = None,
) -> FittedBundle:
    """
    Treina zoo + seleção + ensemble + calibração.
    Não inventa métricas se não houver jogos suficientes — marca not_evaluated.
    """
    t0 = time.time()
    cfg = cfg or load_config()
    max_goals = int(cfg.get("goal_support", 8))
    status = ExperimentStatus(status="not_evaluated")
    status.notes.append("Pipeline iniciado")
    _emit(progress, 0.02, "Preparando dados…")

    matches = matches.sort_values("date", kind="mergesort").reset_index(drop=True)
    matches, _ = update_elo(matches)
    _emit(progress, 0.08, "Gerando features…")
    feats, registry = build_pre_match_features(
        matches,
        rolling_windows=cfg.get("features", {}).get("rolling_windows", [3, 5, 8]),
        ewma_halflives=cfg.get("features", {}).get("ewma_halflives", [3, 5, 10]),
    )

    played = matches.loc[_played_mask(matches)]
    if len(played) < 15:
        status.notes.append("Poucos jogos com placar — champion não avaliado")
        models = build_model_zoo(max_goals, cfg.get("models_enabled"))
        for m in tqdm(models, desc="Fit rápido", leave=False):
            m.fit(matches, feats)
        bundle = FittedBundle(
            models=models,
            model_names=[m.name for m in models],
            weights=np.ones(len(models)) / max(len(models), 1),
            calibrator=Calibrator("none"),
            features_frame=feats,
            matches=matches,
            status=status,
            selected_features=[],
            max_goals=max_goals,
        )
        _save_status(status, cfg)
        _emit(progress, 1.0, "Concluído")
        return bundle

    _emit(progress, 0.12, "Selecionando features…")
    sel, stab = select_features(
        feats.loc[played.index],
        matches.loc[played.index, "home_goals"],
        matches.loc[played.index, "away_goals"],
    )
    status.selected_features = sel

    enabled = cfg.get("models_enabled")
    models = build_model_zoo(max_goals, enabled)
    status.model_names = [m.name for m in models]

    if cfg.get("hpo", {}).get("enabled", True) and run_backtest:
        n_trials = budget_hpo_trials(cfg)
        _emit(progress, 0.18, f"HPO Dixon-Coles ({n_trials} trials)…")

        def obj(params: dict[str, Any]) -> float:
            from prob_ml.models import DixonColesModel

            m = DixonColesModel(max_goals=max_goals, home_advantage=params["ha"])
            idx = played.index.to_numpy()
            cut = int(len(idx) * 0.7)
            tr, te = idx[:cut], idx[cut:]
            if len(te) < 5:
                return 1e6
            m.fit(matches.loc[tr], feats.loc[tr])
            dists = [m.predict_match(matches.loc[i], feats.loc[i]) for i in te]
            met = aggregate_metrics(
                dists,
                matches.loc[te, "home_goals"].to_numpy(),
                matches.loc[te, "away_goals"].to_numpy(),
            )
            return float(met["score_nll"])

        space = [{"ha": float(x)} for x in np.linspace(0.05, 0.45, max(n_trials, 4))]
        best, best_v = run_hpo(obj, space, n_trials=n_trials, use_optuna=False)
        status.notes.append(f"HPO dixon_coles ha={best['ha']:.3f} nll={best_v:.4f}")
        for i, m in enumerate(models):
            if m.name == "dixon_coles":
                from prob_ml.models import DixonColesModel

                models[i] = DixonColesModel(max_goals=max_goals, home_advantage=best["ha"])

    _emit(progress, 0.30, "Ajustando modelos finais…")
    for mi, m in enumerate(tqdm(models, desc="Fit modelos", leave=False)):
        m.fit(matches, feats)
        _emit(
            progress,
            0.30 + 0.15 * ((mi + 1) / max(len(models), 1)),
            f"Fit: {m.name}",
        )

    splits = expanding_season_splits(matches) or []
    if not splits:
        from prob_ml.backtesting import date_expanding_splits

        splits = date_expanding_splits(matches, n_splits=3)

    oof_by_model: list[list[ScoreDistribution]] = [[] for _ in models]
    oof_hg: list[float] = []
    oof_ag: list[float] = []
    oof_probs: list[list[float]] = []
    oof_y: list[int] = []
    model_metrics: dict[str, dict[str, float]] = {}

    if run_backtest and splits:
        _emit(progress, 0.48, f"Backtest OOF ({len(splits)} folds)…")
        for si, sp in enumerate(tqdm(splits, desc="Folds OOF", leave=False)):
            fitted = build_model_zoo(max_goals, enabled)
            for i, m in enumerate(fitted):
                if m.name == "dixon_coles" and models[i].name == "dixon_coles":
                    m.home_advantage = getattr(models[i], "home_advantage", 0.25)
                m.fit(matches.loc[sp.train_idx], feats.loc[sp.train_idx])
            for j in sp.val_idx:
                if pd.isna(matches.loc[j, "home_goals"]):
                    continue
                hg = int(matches.loc[j, "home_goals"])
                ag = int(matches.loc[j, "away_goals"])
                oof_hg.append(hg)
                oof_ag.append(ag)
                row_dists = []
                for mi, m in enumerate(fitted):
                    d = m.predict_match(matches.loc[j], feats.loc[j])
                    oof_by_model[mi].append(d)
                    row_dists.append(d)
                blend = blend_distributions(row_dists)
                ph, pd_, pa = blend.p_1x2()
                oof_probs.append([ph, pd_, pa])
                oof_y.append(0 if hg > ag else (1 if hg == ag else 2))
            _emit(
                progress,
                0.48 + 0.35 * ((si + 1) / max(len(splits), 1)),
                f"Fold OOF {si + 1}/{len(splits)}",
            )

        for mi, m in enumerate(models):
            if not oof_by_model[mi]:
                continue
            model_metrics[m.name] = aggregate_metrics(
                oof_by_model[mi], np.array(oof_hg), np.array(oof_ag)
            )

        if oof_hg:
            _emit(progress, 0.88, "Ensemble + calibração…")
            method = "performance_weighted"
            w = optimize_blend_weights(
                oof_by_model, np.array(oof_hg), np.array(oof_ag), method=method
            )
            cal = fit_temperature(np.asarray(oof_probs), np.asarray(oof_y))
            ens_dists = []
            for j in tqdm(range(len(oof_hg)), desc="Métricas ensemble", leave=False):
                d = blend_distributions(
                    [oof_by_model[m][j] for m in range(len(models))], w
                )
                d = recalibrate_distribution(d, cal)
                ens_dists.append(d)
            ens_met = aggregate_metrics(ens_dists, np.array(oof_hg), np.array(oof_ag))
            status.status = "trained"
            status.champion = "ensemble"
            status.ensemble_method = method
            status.ensemble_weights = {
                models[i].name: float(w[i]) for i in range(len(models))
            }
            status.metrics = {"models": model_metrics, "ensemble": ens_met}
            status.calibration = cal.method
            status.notes.append(f"OOF n={len(oof_hg)}")
        else:
            w = np.ones(len(models)) / len(models)
            cal = Calibrator("none")
            status.notes.append("Sem OOF válido")
            w = np.ones(len(models)) / max(len(models), 1)
            cal = Calibrator("none")
    else:
        w = np.ones(len(models)) / max(len(models), 1)
        cal = Calibrator("none")
        status.notes.append("Backtest desabilitado nesta chamada")

    status.runtime_sec = round(time.time() - t0, 3)
    from prob_ml.data import dataset_fingerprint

    status.dataset_fingerprint = dataset_fingerprint(matches)

    bundle = FittedBundle(
        models=models,
        model_names=[m.name for m in models],
        weights=w if isinstance(w, np.ndarray) else np.asarray(w),
        calibrator=cal if "cal" in dir() else Calibrator("none"),
        features_frame=feats,
        matches=matches,
        status=status,
        selected_features=sel,
        max_goals=max_goals,
    )
    if run_backtest and splits and oof_hg:
        bundle.calibrator = cal
        bundle.weights = w

    _save_bundle_meta(bundle, cfg)
    _emit(progress, 1.0, "Treino concluído")
    return bundle


def predict_fixtures(
    bundle: FittedBundle,
    fixtures: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Prevê partidas sem placar (ou DF fornecido)."""
    matches = bundle.matches if fixtures is None else fixtures
    feats = bundle.features_frame
    rows = []
    pending = matches[matches["home_goals"].isna() | matches["away_goals"].isna()]
    for i, row in pending.iterrows():
        dists = []
        for m in bundle.models:
            fr = feats.loc[i] if i in feats.index else None
            dists.append(m.predict_match(row, fr))
        dist = blend_distributions(dists, bundle.weights)
        dist = recalibrate_distribution(dist, bundle.calibrator)
        ph, pd_, pa = dist.p_1x2()
        eh, ea = dist.expected_goals()
        xph, xpa = dist.expected_points()
        ou, uu = dist.over_under(2.5)
        by, bn = dist.btts()
        rows.append(
            {
                "index": i,
                "round": row.get("round"),
                "date": row.get("date"),
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "xg_home": eh,
                "xg_away": ea,
                "p_home": ph,
                "p_draw": pd_,
                "p_away": pa,
                "xpts_home": xph,
                "xpts_away": xpa,
                "over_25": ou,
                "under_25": uu,
                "btts_yes": by,
                "btts_no": bn,
                "top_scores": dist.top_scores(5),
                "dist": dist,
                "champion": bundle.status.champion or "ensemble_unvalidated",
                "status": bundle.status.status,
            }
        )
    return rows


def simulate_from_bundle(
    bundle: FittedBundle,
    *,
    n_sims: int | None = None,
    cfg: dict[str, Any] | None = None,
) -> tuple[SeasonSimResult, pd.DataFrame]:
    cfg = cfg or load_config()
    n_sims = n_sims or budget_n_sims(cfg)
    matches = bundle.matches
    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    # estado atual
    current: dict[str, dict[str, float]] = {
        t: {"points": 0.0, "wins": 0.0, "gd": 0.0, "gf": 0.0} for t in teams
    }
    for _, r in matches.dropna(subset=["home_goals", "away_goals"]).iterrows():
        h, a = r["home_team"], r["away_team"]
        hg, ag = int(r["home_goals"]), int(r["away_goals"])
        current[h]["gf"] += hg
        current[a]["gf"] += ag
        current[h]["gd"] += hg - ag
        current[a]["gd"] += ag - hg
        if hg > ag:
            current[h]["points"] += 3
            current[h]["wins"] += 1
        elif hg < ag:
            current[a]["points"] += 3
            current[a]["wins"] += 1
        else:
            current[h]["points"] += 1
            current[a]["points"] += 1

    preds = predict_fixtures(bundle)
    fixtures = [(p["home_team"], p["away_team"], p["dist"]) for p in preds]
    res = simulate_season(teams, current, fixtures, n_sims=n_sims, seed=int(cfg.get("monte_carlo", {}).get("seed", 42)))
    return res, result_to_frame(res)


def _save_status(status: ExperimentStatus, cfg: dict[str, Any]) -> None:
    d = artifacts_dir(cfg)
    path = d / "status.json"
    path.write_text(json.dumps(status.to_dict(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _save_bundle_meta(bundle: FittedBundle, cfg: dict[str, Any]) -> None:
    _save_status(bundle.status, cfg)
    d = artifacts_dir(cfg)
    (d / "model_metrics.json").write_text(
        json.dumps(bundle.status.metrics, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (d / "selected_features.json").write_text(
        json.dumps(bundle.selected_features, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_status(cfg: dict[str, Any] | None = None) -> ExperimentStatus:
    cfg = cfg or load_config()
    path = artifacts_dir(cfg) / "status.json"
    if not path.exists():
        return ExperimentStatus(status="not_evaluated")
    data = json.loads(path.read_text(encoding="utf-8"))
    return ExperimentStatus(**{k: data.get(k) for k in ExperimentStatus.__dataclass_fields__})
