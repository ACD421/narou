from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ..config import LABELED_DIR
from ..schema import Job
from ..utils import parse_iso_datetime
from .classifier import FraudClassifier
from .features import FraudFeatures, extract_batch
from .heuristics import score_heuristics


class ScaledLogistic:
    def __init__(self, scaler, model):
        self._scaler = scaler
        self._model = model

    def predict_proba(self, X):
        return self._model.predict_proba(self._scaler.transform(X))


@dataclass
class TrainingData:
    features: list[FraudFeatures]
    labels: list[int]
    source: str

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        X = np.array([f.to_vector() for f in self.features], dtype=np.float32)
        y = np.array(self.labels, dtype=np.int32)
        return X, y


def weak_label_from_heuristics(
    jobs: list[Job],
    flag_threshold: float = 0.5,
    safe_threshold: float = 0.15,
) -> TrainingData:
    feats = extract_batch(jobs)
    X_feats: list[FraudFeatures] = []
    y: list[int] = []
    for job, f in zip(jobs, feats):
        h = score_heuristics(f)
        if h.score >= flag_threshold:
            X_feats.append(f)
            y.append(1)
        elif h.score <= safe_threshold:
            X_feats.append(f)
            y.append(0)
    return TrainingData(features=X_feats, labels=y, source="weak_labels")


def load_labeled_csv(path: str | Path) -> TrainingData:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    rows = []
    with open(p, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return TrainingData(features=[], labels=[], source=str(p))

    columns = set(rows[0].keys())
    fraud_col = None
    for candidate in ("fraudulent", "fraud", "fake", "label"):
        if candidate in columns:
            fraud_col = candidate
            break
    if fraud_col is None:
        raise ValueError(f"no label column in {columns}")

    title_col = "title" if "title" in columns else None
    company_col = (
        "company_profile" if "company_profile" in columns
        else "company" if "company" in columns else None
    )
    desc_col = None
    for candidate in ("description", "job_description", "requirements", "body"):
        if candidate in columns:
            desc_col = candidate
            break
    if desc_col is None:
        raise ValueError(f"no description column in {columns}")

    jobs: list[Job] = []
    labels: list[int] = []
    for i, row in enumerate(rows):
        try:
            label = int(row[fraud_col])
        except (ValueError, TypeError):
            continue
        desc = row.get(desc_col, "") or ""
        title = row.get(title_col, "") if title_col else ""
        company = row.get(company_col, "") if company_col else "unknown"
        if not desc:
            continue
        posted = parse_iso_datetime(row.get("posted_at"))
        jobs.append(Job(
            job_id=str(i),
            source="csv",
            company=company or "unknown",
            title=title or "Untitled",
            location=row.get("location", "") or "",
            department=row.get("department") or None,
            description=desc,
            url="",
            posted_at=posted,
            updated_at=None,
        ))
        labels.append(1 if label else 0)

    feats = extract_batch(jobs)
    return TrainingData(features=feats, labels=labels, source=str(p))


def train_models(data: TrainingData) -> FraudClassifier:
    if len(data.features) < 20:
        raise ValueError(f"not enough samples: {len(data.features)}")

    X, y = data.as_arrays()
    pos = int(y.sum())
    neg = int(len(y) - pos)
    if pos == 0 or neg == 0:
        raise ValueError(f"need both classes: pos={pos} neg={neg}")

    stratify = y if min(pos, neg) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    logistic = LogisticRegression(
        max_iter=1000, class_weight="balanced", solver="lbfgs"
    )
    logistic.fit(X_train_s, y_train)

    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    lp = logistic.predict(X_test_s)
    rp = rf.predict(X_test)
    metrics = {
        "source": data.source,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "pos_rate": float(pos / len(y)),
    }
    for name, preds, probs_fn in [
        ("logistic", lp, lambda: logistic.predict_proba(X_test_s)[:, 1]),
        ("random_forest", rp, lambda: rf.predict_proba(X_test)[:, 1]),
    ]:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="binary", zero_division=0
        )
        try:
            auc = float(roc_auc_score(y_test, probs_fn()))
        except ValueError:
            auc = 0.0
        metrics[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": auc,
        }

    classifier = FraudClassifier(
        logistic=ScaledLogistic(scaler, logistic),
        random_forest=rf,
        feature_names=FraudFeatures.feature_names(),
        trained_on=len(data.features),
        metrics=metrics,
    )
    return classifier


def find_labeled_csv() -> Path | None:
    if not LABELED_DIR.exists():
        return None
    for p in LABELED_DIR.glob("*.csv"):
        return p
    return None
