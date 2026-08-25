"""Monitoramento / drift (preparação)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DriftReport:
    status: str = "not_evaluated"
    feature_drift: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "feature_drift": self.feature_drift,
            "notes": self.notes,
        }


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index simples."""
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < 20 or len(actual) < 20:
        return float("nan")
    qs = np.linspace(0, 100, bins + 1)
    cuts = np.unique(np.percentile(expected, qs))
    if len(cuts) < 3:
        return float("nan")
    e_counts = np.histogram(expected, bins=cuts)[0].astype(float) + 1e-6
    a_counts = np.histogram(actual, bins=cuts)[0].astype(float) + 1e-6
    e_pct = e_counts / e_counts.sum()
    a_pct = a_counts / a_counts.sum()
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))
