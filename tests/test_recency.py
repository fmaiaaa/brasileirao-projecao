"""Testes de janela histórica e pesos por recência."""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from recency import (
    attach_sample_weights,
    cutoff_date,
    exponential_decay_weight,
    filter_jogos_by_recency,
    filter_matches_dataframe,
    weighted_mean,
    wls_lstsq,
)
from brasileirao_projecao_core import Jogo, media_pts_jogo


def test_cutoff_three_years():
    ref = date(2026, 8, 31)
    assert cutoff_date(ref=ref, years=3) == date(2024, 1, 1)


def test_exponential_decay_old_games_much_lighter():
    w_recent = exponential_decay_weight(7, half_life_days=120)
    w_old = exponential_decay_weight(730, half_life_days=120)
    assert w_recent > 0.9
    assert w_old < 0.02
    assert w_recent > 50 * w_old


def test_filter_matches_dataframe():
    ref = date(2026, 6, 1)
    df = pd.DataFrame(
        {
            "date": ["2020-01-01", "2024-03-01", "2026-01-01"],
            "season": [2020, 2024, 2026],
            "home_goals": [1, 2, 1],
        }
    )
    out = filter_matches_dataframe(df, years=3, ref=ref)
    assert len(out) == 2
    assert 2020 not in set(out["season"])


def test_weighted_mean_and_wls():
    w = np.array([1.0, 1.0, 0.01])
    assert weighted_mean(np.array([0.0, 0.0, 100.0]), w) < 1.0
    X = np.column_stack([np.ones(3), np.arange(3, dtype=float)])
    y = np.array([1.0, 2.0, 3.0])
    coef = wls_lstsq(X, y, w)
    assert coef.shape == (2,)


def test_media_pts_jogo_weighted_by_date():
    ref = date(2026, 8, 31)
    old = (ref - timedelta(days=800)).isoformat()
    recent = (ref - timedelta(days=14)).isoformat()
    jogos = [
        Jogo(1, old, "", "A", "B", "0 x 3"),
        Jogo(2, recent, "", "A", "C", "3 x 0"),
    ]
    m = media_pts_jogo(jogos, "A", 1, 2, "simples")
    assert m["geral"] > 1.5


def test_filter_jogos_by_recency():
    ref = date(2026, 8, 31)
    jogos = [
        Jogo(1, "2020-05-01", "", "X", "Y", "1 x 0"),
        Jogo(2, "2025-05-01", "", "X", "Z", "1 x 0"),
    ]
    out = filter_jogos_by_recency(jogos, years=3, ref=ref)
    assert len(out) == 1
    assert out[0].vis == "Z"


def test_attach_sample_weights_column():
    ref = date(2026, 8, 31)
    df = pd.DataFrame({"date": [ref - timedelta(days=d) for d in (0, 120, 480)]})
    out = attach_sample_weights(df, ref=ref, half_life_days=120)
    assert "sample_weight" in out.columns
    assert out["sample_weight"].iloc[0] > out["sample_weight"].iloc[-1]
