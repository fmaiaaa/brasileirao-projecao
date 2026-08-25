"""Carregamento de configuração YAML do pipeline probabilístico."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = _ROOT / "config" / "prob_ml.yaml"
DEFAULT_SCHEMA_MAP_PATH = _ROOT / "config" / "schema_map.yaml"


def load_yaml(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config não encontrada: {p}")
    with p.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML inválido (esperado mapping): {p}")
    return data


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    return load_yaml(path or DEFAULT_CONFIG_PATH)


def load_schema_map(path: Path | str | None = None) -> dict[str, Any]:
    return load_yaml(path or DEFAULT_SCHEMA_MAP_PATH)


def budget_n_sims(cfg: dict[str, Any]) -> int:
    budget = str(cfg.get("compute_budget", "fast")).lower()
    mc = cfg.get("monte_carlo", {})
    key = {
        "fast": "n_sims_fast",
        "standard": "n_sims_standard",
        "full": "n_sims_full",
    }.get(budget, "n_sims_fast")
    return int(mc.get(key, 20_000))


def budget_hpo_trials(cfg: dict[str, Any]) -> int:
    budget = str(cfg.get("compute_budget", "fast")).lower()
    hpo = cfg.get("hpo", {})
    key = {
        "fast": "n_trials_fast",
        "standard": "n_trials_standard",
        "full": "n_trials_full",
    }.get(budget, "n_trials_fast")
    return int(hpo.get(key, 8))


def artifacts_dir(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_config()
    d = _ROOT / str(cfg.get("artifacts_dir", "artifacts/prob_ml"))
    d.mkdir(parents=True, exist_ok=True)
    return d
