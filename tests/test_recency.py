"""Testes de janela de 38 rodadas e pesos por distância de rodada."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from recency import (
    W_MAX,
    W_SEASON_FLOOR,
    attach_sample_weights,
    detect_promoted_teams,
    elastic_net_lstsq,
    exponential_decay_rounds,
    filter_jogos_by_round_window,
    filter_matches_dataframe,
    importance_weight_by_rounds_ago,
    load_regression_settings,
    weight_for_jogo,
    weighted_mean,
    wls_lstsq,
)
from brasileirao_projecao_core import Jogo, media_pts_jogo


def _serie_a_row(season: int, round_num: int, home: str, away: str) -> dict:
    return {
        "date": f"{season}-03-{min(round_num, 28):02d}",
        "season": season,
        "round": round_num,
        "competition": "serie_a",
        "home_team": home,
        "away_team": away,
        "home_goals": 1,
        "away_goals": 0,
    }


def test_importance_weight_scheme():
    """Últimas 5 rodadas 100→90%; depois 90→25%; anos passados 25→0%."""
    w0 = importance_weight_by_rounds_ago(0)
    w4 = importance_weight_by_rounds_ago(4)
    w5 = importance_weight_by_rounds_ago(5, rounds_ago_season_start=37)
    w_old = importance_weight_by_rounds_ago(37, rounds_ago_season_start=37)
    w_past = importance_weight_by_rounds_ago(0, is_past_season=True, past_year_fraction=0.0)
    w_past_old = importance_weight_by_rounds_ago(0, is_past_season=True, past_year_fraction=1.0)
    assert w0 == W_MAX
    assert 0.89 <= w4 <= 0.91
    assert 0.88 <= w5 <= 0.92
    assert abs(w_old - W_SEASON_FLOOR) < 0.02
    assert abs(w_past - 0.25) < 0.02
    assert w_past_old == 0.0
    assert w0 > w_old > w_past_old


def test_exponential_decay_legacy():
    w_recent = exponential_decay_rounds(1, half_life_rounds=12)
    w_old = exponential_decay_rounds(37, half_life_rounds=12)
    assert w_recent > w_old


def test_filter_matches_last_38_rounds():
    rows = []
    for r in range(1, 45):
        rows.append(_serie_a_row(2025, r, "A", "B"))
    df = pd.DataFrame(rows)
    out = filter_matches_dataframe(df, history_rounds=38)
    rounds = sorted(out["round"].unique())
    assert len(rounds) == 38
    assert rounds[0] == 7
    assert rounds[-1] == 44


def test_promoted_teams_get_serie_b_history():
    sa_2025 = [_serie_a_row(2025, r, "TimeA", "TimeB") for r in range(1, 39)]
    sa_2026 = [_serie_a_row(2026, r, "TimeA", "NovoTime") for r in range(1, 10)]
    sb = [
        {
            "date": "2025-08-01",
            "season": 2025,
            "round": 20,
            "competition": "serie_b",
            "home_team": "NovoTime",
            "away_team": "Outro",
            "home_goals": 2,
            "away_goals": 1,
        }
    ]
    df = pd.DataFrame(sa_2025 + sa_2026 + sb)
    promoted = detect_promoted_teams(df, calendar_teams={"TimeA", "NovoTime"}, current_season=2026)
    assert "NovoTime" in promoted
    out = filter_matches_dataframe(
        df, history_rounds=38, calendar_teams={"TimeA", "NovoTime"}, current_season=2026
    )
    sb_out = out[out["competition"] == "serie_b"]
    assert len(sb_out) == 1
    assert sb_out.iloc[0]["home_team"] == "NovoTime"


def test_weighted_mean_and_wls():
    w = np.array([1.0, 1.0, 0.01])
    assert weighted_mean(np.array([0.0, 0.0, 100.0]), w) < 1.0
    X = np.column_stack([np.ones(3), np.arange(3, dtype=float)])
    y = np.array([1.0, 2.0, 3.0])
    coef = wls_lstsq(X, y, w)
    assert coef.shape == (2,)


def test_media_pts_jogo_weighted_by_round():
    jogos = [Jogo(r, "", "", "A", "B", "0 x 3") for r in range(1, 50)]
    jogos.append(Jogo(50, "", "", "A", "C", "3 x 0"))
    m = media_pts_jogo(jogos, "A", 1, 50, "simples")
    # Com decaimento, vitória recente pesa mais que a média uniforme (3/38)
    assert m["geral"] > 3.0 / 38
    w_new = weight_for_jogo(Jogo(50, "", "", "A", "C", "3 x 0"), r_latest=50)
    w_old = weight_for_jogo(Jogo(13, "", "", "A", "B", "0 x 3"), r_latest=50)
    assert w_new > w_old
    assert w_new >= 0.99


def test_filter_jogos_by_round_window():
    jogos = [Jogo(r, "", "", "X", "Y", "1 x 0") for r in range(1, 50)]
    out = filter_jogos_by_round_window(jogos, n_rounds=38)
    played = [j.r for j in out if j.jogado]
    assert min(played) == 12
    assert max(played) == 49


def test_attach_sample_weights_by_round():
    df = pd.DataFrame([_serie_a_row(2026, r, "A", "B") for r in (38, 30, 1)])
    out = attach_sample_weights(df, half_life_rounds=12)
    assert "sample_weight" in out.columns
    w38 = out.loc[out["round"] == 38, "sample_weight"].iloc[0]
    w1 = out.loc[out["round"] == 1, "sample_weight"].iloc[0]
    assert w38 >= 0.99
    assert w1 <= 0.30
    assert w38 > w1


def test_elastic_net_lstsq_returns_coef_vector():
    X = np.column_stack([np.ones(40), np.linspace(1, 5, 40), np.random.default_rng(0).normal(size=40)])
    y = 2.0 + 0.5 * X[:, 1] + np.random.default_rng(1).normal(scale=0.2, size=40)
    coef = elastic_net_lstsq(X, y)
    assert coef.shape == (3,)
    assert coef[1] > 0


def test_weight_for_jogo():
    j_old = Jogo(1, "", "", "A", "B", "1 x 0")
    j_new = Jogo(20, "", "", "A", "C", "1 x 0")
    assert weight_for_jogo(j_new, r_latest=20) > weight_for_jogo(j_old, r_latest=20)


def test_indicador_casa_dummy():
    from brasileirao_projecao_core import _indicador_casa_jogo

    j_casa = Jogo(1, "2026-01-01", "", "Flamengo", "Vasco", "1 x 0")
    j_fora = Jogo(2, "2026-01-08", "", "Vasco", "Flamengo", "0 x 1")
    assert _indicador_casa_jogo(j_casa, "Flamengo") == 1.0
    assert _indicador_casa_jogo(j_fora, "Flamengo") == 0.0
