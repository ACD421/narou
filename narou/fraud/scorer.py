from __future__ import annotations

from dataclasses import dataclass, field

from ..config import FRAUD_FLAG_THRESHOLD
from ..schema import Job
from .classifier import FraudClassifier, load_classifier
from .features import FraudFeatures, extract_batch
from .heuristics import HeuristicResult, score_heuristics

__all__ = ["FraudReport", "score_job", "score_jobs"]


@dataclass
class FraudReport:
    job_uid: str
    score: float
    flagged: bool
    heuristic_score: float
    classifier_score: float
    reasons: list[str] = field(default_factory=list)
    features: FraudFeatures | None = None

    def to_dict(self) -> dict:
        return {
            "job_uid": self.job_uid,
            "score": self.score,
            "flagged": self.flagged,
            "heuristic_score": self.heuristic_score,
            "classifier_score": self.classifier_score,
            "reasons": self.reasons,
            "features": self.features.to_dict() if self.features else None,
        }


def _combine(heuristic: float, classifier_p: float, has_classifier: bool) -> float:
    if not has_classifier:
        return heuristic
    return 0.6 * heuristic + 0.4 * classifier_p


def score_job(
    job: Job,
    features: FraudFeatures,
    classifier: FraudClassifier | None = None,
    threshold: float = FRAUD_FLAG_THRESHOLD,
) -> FraudReport:
    classifier = classifier or FraudClassifier()
    h = score_heuristics(features)
    c = classifier.predict_proba(features) if classifier.is_trained() else 0.0
    combined = _combine(h.score, c, classifier.is_trained())
    return FraudReport(
        job_uid=job.uid,
        score=combined,
        flagged=combined >= threshold,
        heuristic_score=h.score,
        classifier_score=c,
        reasons=h.reasons,
        features=features,
    )


def score_jobs(
    jobs: list[Job],
    classifier: FraudClassifier | None = None,
    threshold: float = FRAUD_FLAG_THRESHOLD,
    dedup_map: dict[str, dict] | None = None,
) -> list[FraudReport]:
    if not jobs:
        return []
    classifier = classifier or load_classifier()
    feats = extract_batch(jobs, dedup_map=dedup_map)
    by_uid = {f.job_uid: f for f in feats}

    ordered_feats: list[FraudFeatures] = []
    ordered_jobs: list[Job] = []
    for j in jobs:
        f = by_uid.get(j.uid)
        if f is None:
            continue
        ordered_jobs.append(j)
        ordered_feats.append(f)

    # Batched classifier call -- one big matrix multiply is massively faster
    # than N small predict_proba calls on sklearn tree ensembles.
    classifier_probs = (
        classifier.predict_batch(ordered_feats)
        if classifier.is_trained()
        else [0.0] * len(ordered_feats)
    )

    has_clf = classifier.is_trained()
    out: list[FraudReport] = []
    for job, f, cp in zip(ordered_jobs, ordered_feats, classifier_probs):
        h = score_heuristics(f)
        combined = 0.6 * h.score + 0.4 * cp if has_clf else h.score
        out.append(FraudReport(
            job_uid=job.uid,
            score=combined,
            flagged=combined >= threshold,
            heuristic_score=h.score,
            classifier_score=cp,
            reasons=h.reasons,
            features=f,
        ))
    return out
