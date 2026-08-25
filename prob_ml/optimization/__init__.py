"""HPO leve (grid/random sem Optuna obrigatório; Optuna se disponível)."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np


def run_hpo(
    objective: Callable[[dict[str, Any]], float],
    search_space: list[dict[str, Any]],
    *,
    n_trials: int = 8,
    seed: int = 42,
    use_optuna: bool = True,
) -> tuple[dict[str, Any], float]:
    """Minimiza objective. Retorna (best_params, best_value)."""
    if use_optuna:
        try:
            import optuna

            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def _obj(trial: optuna.Trial) -> float:
                # sample from discrete space by index
                params = search_space[trial.suggest_int("idx", 0, len(search_space) - 1)]
                return float(objective(params))

            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
            study.optimize(_obj, n_trials=min(n_trials, len(search_space)), show_progress_bar=False)
            best = search_space[study.best_params["idx"]]
            return best, float(study.best_value)
        except Exception:
            pass

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(search_space))[:n_trials]
    best_p, best_v = search_space[0], float("inf")
    for i in order:
        p = search_space[int(i)]
        v = float(objective(p))
        if v < best_v:
            best_v, best_p = v, p
    return best_p, best_v
