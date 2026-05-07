from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from ..config import FRAUD_MODEL_PATH
from .features import FraudFeatures


@dataclass
class FraudClassifier:
    logistic: object | None = None
    random_forest: object | None = None
    feature_names: list[str] | None = None
    trained_on: int = 0
    metrics: dict | None = None

    def is_trained(self) -> bool:
        return self.logistic is not None or self.random_forest is not None

    def predict_proba(self, features: FraudFeatures) -> float:
        if not self.is_trained():
            return 0.0
        vec = np.array([features.to_vector()], dtype=np.float32)
        return float(self._predict_matrix(vec)[0])

    def predict_batch(self, features_list: list[FraudFeatures]) -> list[float]:
        if not features_list or not self.is_trained():
            return [0.0] * len(features_list)
        mat = np.array([f.to_vector() for f in features_list], dtype=np.float32)
        return [float(x) for x in self._predict_matrix(mat)]

    def _predict_matrix(self, mat: np.ndarray) -> np.ndarray:
        probs = []
        if self.logistic is not None:
            probs.append(self.logistic.predict_proba(mat)[:, 1])
        if self.random_forest is not None:
            probs.append(self.random_forest.predict_proba(mat)[:, 1])
        if not probs:
            return np.zeros(mat.shape[0], dtype=np.float32)
        return np.mean(probs, axis=0)

    def save(self, path: str | Path = FRAUD_MODEL_PATH) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "logistic": self.logistic,
                "random_forest": self.random_forest,
                "feature_names": self.feature_names,
                "trained_on": self.trained_on,
                "metrics": self.metrics,
            },
            p,
        )


def load_classifier(path: str | Path = FRAUD_MODEL_PATH) -> FraudClassifier:
    p = Path(path)
    if not p.exists():
        return FraudClassifier()
    try:
        data = joblib.load(p)
    except Exception:
        return FraudClassifier()
    return FraudClassifier(
        logistic=data.get("logistic"),
        random_forest=data.get("random_forest"),
        feature_names=data.get("feature_names"),
        trained_on=data.get("trained_on", 0),
        metrics=data.get("metrics"),
    )
