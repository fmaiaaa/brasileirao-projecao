"""Interface comum de modelos de placar."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from prob_ml.models.score_matrix import ScoreDistribution


class ScoreModel(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, matches: pd.DataFrame, features: pd.DataFrame | None = None) -> "ScoreModel":
        ...

    @abstractmethod
    def predict_match(
        self, row: pd.Series, features_row: pd.Series | None = None
    ) -> ScoreDistribution:
        ...

    def predict_many(
        self, matches: pd.DataFrame, features: pd.DataFrame | None = None
    ) -> list[ScoreDistribution]:
        feats = features if features is not None else matches
        out = []
        for i in matches.index:
            fr = feats.loc[i] if i in feats.index else None
            out.append(self.predict_match(matches.loc[i], fr))
        return out

    def get_params(self) -> dict[str, Any]:
        return {"name": self.name}
