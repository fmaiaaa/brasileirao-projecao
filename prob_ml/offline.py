"""Persistência offline do pipeline probabilístico (treino semanal)."""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from prob_ml.config import artifacts_dir, load_config
from prob_ml.ensemble import blend_distributions
from prob_ml.calibration import recalibrate_distribution
from prob_ml.pipeline import FittedBundle

logger = logging.getLogger(__name__)

BUNDLE_NAME = "champion_bundle.joblib"
FORECASTS_NAME = "match_forecasts.csv"
STANDINGS_NAME = "standings_sim.csv"
CALENDAR_PROJ_NAME = "calendar_projecoes.csv"
META_NAME = "offline_meta.json"

# Calendário → FPT (quando o nome difere)
_TEAM_ALIASES = {
    "atletico-mg": "Atletico-MG",
    "atlético-mg": "Atletico-MG",
    "botafogo": "Botafogo RJ",
    "chapecoense": "Chapecoense-SC",
    "flamengo": "Flamengo RJ",
    "gremio": "Gremio",
    "grêmio": "Gremio",
    "sao paulo": "Sao Paulo",
    "são paulo": "Sao Paulo",
    "vasco": "Vasco",
    "vitoria": "Vitoria",
    "vitória": "Vitoria",
}


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def normalize_team_key(name: str) -> str:
    s = _strip_accents(str(name)).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def map_team_to_fpt(name: str, fpt_teams: set[str]) -> str:
    """Mapeia nome do calendário para o rótulo usado na base FPT."""
    raw = str(name).strip()
    if raw in fpt_teams:
        return raw
    key = normalize_team_key(raw)
    if key in _TEAM_ALIASES and _TEAM_ALIASES[key] in fpt_teams:
        return _TEAM_ALIASES[key]
    # match por chave normalizada
    by_key = {normalize_team_key(t): t for t in fpt_teams}
    if key in by_key:
        return by_key[key]
    # contém
    for k, t in by_key.items():
        if key in k or k in key:
            return t
    return raw


def artifact_paths(cfg: dict[str, Any] | None = None) -> dict[str, Path]:
    d = artifacts_dir(cfg)
    return {
        "dir": d,
        "bundle": d / BUNDLE_NAME,
        "forecasts": d / FORECASTS_NAME,
        "standings": d / STANDINGS_NAME,
        "calendar": d / CALENDAR_PROJ_NAME,
        "meta": d / META_NAME,
        "status": d / "status.json",
    }


def save_offline_artifacts(
    bundle: FittedBundle,
    *,
    forecasts: list[dict[str, Any]] | None = None,
    standings: pd.DataFrame | None = None,
    calendar_proj: pd.DataFrame | None = None,
    cfg: dict[str, Any] | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Path]:
    cfg = cfg or load_config()
    paths = artifact_paths(cfg)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    joblib.dump(bundle, paths["bundle"])
    logger.info("Bundle salvo: %s", paths["bundle"])

    if forecasts is not None:
        rows = []
        for p in forecasts:
            rows.append(
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
            )
        pd.DataFrame(rows).to_csv(paths["forecasts"], index=False, encoding="utf-8-sig")

    if standings is not None and not standings.empty:
        standings.to_csv(paths["standings"], index=False, encoding="utf-8-sig")

    if calendar_proj is not None and not calendar_proj.empty:
        calendar_proj.to_csv(paths["calendar"], index=False, encoding="utf-8-sig")

    meta = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "status": bundle.status.status,
        "champion": bundle.status.champion,
        "fingerprint": bundle.status.dataset_fingerprint,
        "ensemble_weights": bundle.status.ensemble_weights,
        "metrics": bundle.status.metrics,
        "runtime_sec": bundle.status.runtime_sec,
        "n_matches": int(len(bundle.matches)),
        **(extra_meta or {}),
    }
    paths["meta"].write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return paths


def _latest_team_features(feats: pd.DataFrame, team_fpt: str, as_home: bool) -> dict[str, Any]:
    """Última linha de features conhecida do time (prefixo home_/away_)."""
    prefix = "home_" if as_home else "away_"
    team_col = "home_team" if as_home else "away_team"
    sub = feats[feats[team_col].astype(str) == team_fpt]
    if sub.empty:
        # tenta o outro lado
        other = "away_team" if as_home else "home_team"
        other_pref = "away_" if as_home else "home_"
        sub2 = feats[feats[other].astype(str) == team_fpt]
        if sub2.empty:
            return {}
        row = sub2.iloc[-1]
        out = {}
        for c in feats.columns:
            if c.startswith(other_pref):
                out[prefix + c[len(other_pref) :]] = row[c]
        return out
    row = sub.iloc[-1]
    return {c: row[c] for c in feats.columns if c.startswith(prefix)}


def forecast_calendar_games(bundle: FittedBundle, jogos: list[Any]) -> list[dict[str, Any]]:
    """
    Prevê jogos pendentes do calendário do app com o champion offline.
    Faz o de-para de nomes Calendário ↔ FPT.
    """
    fpt_teams = set(bundle.matches["home_team"].astype(str)) | set(
        bundle.matches["away_team"].astype(str)
    )
    feats = bundle.features_frame
    out: list[dict[str, Any]] = []

    for j in jogos:
        if getattr(j, "jogado", False):
            continue
        h_cal, a_cal = str(j.mand), str(j.vis)
        h = map_team_to_fpt(h_cal, fpt_teams)
        a = map_team_to_fpt(a_cal, fpt_teams)
        row = pd.Series({"home_team": h, "away_team": a, "round": j.r})
        feat_row = pd.Series(dtype=object)
        feat_row["home_team"] = h
        feat_row["away_team"] = a
        for k, v in _latest_team_features(feats, h, True).items():
            feat_row[k] = v
        for k, v in _latest_team_features(feats, a, False).items():
            feat_row[k] = v
        if "league_avg_goals_pre" in feats.columns:
            feat_row["league_avg_goals_pre"] = feats["league_avg_goals_pre"].dropna().iloc[-1]

        dists = []
        for m in bundle.models:
            dists.append(m.predict_match(row, feat_row))
        dist = blend_distributions(dists, bundle.weights)
        dist = recalibrate_distribution(dist, bundle.calibrator)
        ph, pd_, pa = dist.p_1x2()
        eh, ea = dist.expected_goals()
        xph, xpa = dist.expected_points()
        ou, uu = dist.over_under(2.5)
        by, bn = dist.btts()
        out.append(
            {
                "round": j.r,
                "date": getattr(j, "data", None),
                "home_team": h_cal,
                "away_team": a_cal,
                "home_team_fpt": h,
                "away_team_fpt": a,
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
                "champion": bundle.status.champion or "ensemble",
                "status": bundle.status.status,
                "dist": dist,
            }
        )
    return out


def load_bundle(cfg: dict[str, Any] | None = None) -> FittedBundle | None:
    paths = artifact_paths(cfg)
    if not paths["bundle"].exists():
        return None
    try:
        return joblib.load(paths["bundle"])
    except Exception as e:
        logger.warning("Falha ao carregar bundle: %s", e)
        return None


def load_calendar_projecoes(cfg: dict[str, Any] | None = None) -> pd.DataFrame | None:
    paths = artifact_paths(cfg)
    if not paths["calendar"].exists():
        return None
    return pd.read_csv(paths["calendar"])


def load_standings(cfg: dict[str, Any] | None = None) -> pd.DataFrame | None:
    paths = artifact_paths(cfg)
    if not paths["standings"].exists():
        return None
    return pd.read_csv(paths["standings"])


def load_forecasts(cfg: dict[str, Any] | None = None) -> pd.DataFrame | None:
    paths = artifact_paths(cfg)
    if not paths["forecasts"].exists():
        return None
    return pd.read_csv(paths["forecasts"])


def load_offline_meta(cfg: dict[str, Any] | None = None) -> dict[str, Any] | None:
    paths = artifact_paths(cfg)
    if not paths["meta"].exists():
        return None
    return json.loads(paths["meta"].read_text(encoding="utf-8"))


def offline_ready(cfg: dict[str, Any] | None = None) -> bool:
    """App: basta CSV semanal de projeções (bundle opcional p/ refresh MC)."""
    paths = artifact_paths(cfg)
    return paths["calendar"].exists() or paths["forecasts"].exists()
