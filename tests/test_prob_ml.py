"""Testes do pipeline probabilístico."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prob_ml.data import (
    make_synthetic_matches,
    parse_drive_file_id,
    dataset_fingerprint,
    infer_and_map_schema,
    normalize_matches,
)
from prob_ml.features import assert_no_future_leakage, build_pre_match_features
from prob_ml.models.score_matrix import (
    assert_valid_distribution,
    dixon_coles_matrix,
    independent_poisson_matrix,
)
from prob_ml.models import DixonColesModel, LeagueMeanModel
from prob_ml.pipeline import train_pipeline, predict_fixtures
from prob_ml.simulation import simulate_season
from prob_ml.ratings import update_elo


def test_parse_drive_id():
    assert parse_drive_file_id("1AbCdefghijklmnopqr") == "1AbCdefghijklmnopqr"
    url = "https://drive.google.com/file/d/1AbCdefghijklmnopqr/view?usp=sharing"
    assert parse_drive_file_id(url) == "1AbCdefghijklmnopqr"


def test_score_matrix_valid():
    d = independent_poisson_matrix(1.4, 1.1, max_goals=6)
    assert_valid_distribution(d)
    ph, pd, pa = d.p_1x2()
    assert abs(ph + pd + pa - 1) < 1e-6
    d2 = dixon_coles_matrix(1.4, 1.1, rho=-0.1, max_goals=6)
    assert_valid_distribution(d2)


def test_schema_mapping():
    raw = pd.DataFrame(
        {
            "Date": ["2024-01-01"],
            "Home": ["A"],
            "Away": ["B"],
            "Home_Score": [2],
            "Away_Score": [1],
        }
    )
    mapped, used = infer_and_map_schema(raw)
    assert "home_goals" in mapped.columns
    assert used["home_team"] == "Home"


def test_leakage_features():
    df = make_synthetic_matches(n_teams=8, n_rounds=8, seed=2)

    def builder(m):
        return build_pre_match_features(m, rolling_windows=[3, 5], ewma_halflives=[3])

    assert_no_future_leakage(df, builder, mutate_from_idx=len(df) // 2)


def test_elo_pre_match():
    df = make_synthetic_matches(seed=3)
    out, state = update_elo(df)
    assert "elo_home_pre" in out.columns
    assert len(state.ratings) > 0


def test_train_and_predict_synthetic():
    df = make_synthetic_matches(n_teams=8, n_rounds=10, seed=4)
    bundle = train_pipeline(df, run_backtest=True)
    assert bundle.models
    preds = predict_fixtures(bundle)
    # todas as partidas têm placar no sintético → preds pode ser vazio
    # crie pendentes
    df2 = df.copy()
    df2.loc[df2.index[-3:], "home_goals"] = np.nan
    df2.loc[df2.index[-3:], "away_goals"] = np.nan
    bundle2 = train_pipeline(df2, run_backtest=False)
    preds2 = predict_fixtures(bundle2)
    assert len(preds2) == 3
    for p in preds2:
        assert abs(p["p_home"] + p["p_draw"] + p["p_away"] - 1) < 1e-5


def test_monte_carlo_repro():
    teams = ["A", "B", "C", "D"]
    current = {t: {"points": 0, "wins": 0, "gd": 0, "gf": 0} for t in teams}
    dist = independent_poisson_matrix(1.2, 1.0, max_goals=5)
    fixtures = [("A", "B", dist), ("C", "D", dist), ("A", "C", dist), ("B", "D", dist)]
    r1 = simulate_season(teams, current, fixtures, n_sims=2000, seed=7)
    r2 = simulate_season(teams, current, fixtures, n_sims=2000, seed=7)
    assert np.allclose(r1.p_champion, r2.p_champion)
    assert abs(r1.p_champion.sum() - 1) < 1e-6


def test_fingerprint_stable():
    df = make_synthetic_matches(seed=5)
    a = dataset_fingerprint(df)
    b = dataset_fingerprint(df.copy())
    assert a == b


def test_model_fit_predict():
    df = make_synthetic_matches(n_rounds=8, seed=6)
    m = DixonColesModel(max_goals=5)
    m.fit(df)
    d = m.predict_match(df.iloc[0])
    assert_valid_distribution(d)
    lm = LeagueMeanModel()
    lm.fit(df)
    assert_valid_distribution(lm.predict_match(df.iloc[0]))
