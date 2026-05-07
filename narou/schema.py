from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _norm_hash(text: str) -> str:
    squashed = " ".join(text.lower().split())
    return hashlib.sha256(squashed.encode("utf-8")).hexdigest()[:16]


@dataclass
class Job:
    job_id: str
    source: str
    company: str
    title: str
    location: str
    department: str | None
    description: str
    url: str
    posted_at: datetime | None
    updated_at: datetime | None
    fetched_at: datetime = field(default_factory=_utcnow)
    description_hash: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.description_hash and self.description:
            self.description_hash = _norm_hash(self.description)

    @property
    def days_active(self) -> int | None:
        if self.posted_at is None:
            return None
        delta = _utcnow() - self.posted_at
        return max(0, delta.days)

    @property
    def uid(self) -> str:
        return f"{self.source}:{self.job_id}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["posted_at"] = self.posted_at.isoformat() if self.posted_at else None
        d["updated_at"] = self.updated_at.isoformat() if self.updated_at else None
        d["fetched_at"] = self.fetched_at.isoformat()
        d["days_active"] = self.days_active
        d["uid"] = self.uid
        return d


@dataclass
class Resume:
    raw_text: str
    sections: dict[str, str]
    contact: dict[str, str]
    skills: list[str]
    source_filename: str
    parsed_at: datetime = field(default_factory=_utcnow)

    def summary_text(self) -> str:
        parts = []
        for key in ("summary", "experience", "skills", "education"):
            if key in self.sections and self.sections[key]:
                parts.append(self.sections[key])
        return "\n\n".join(parts) if parts else self.raw_text

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "sections": self.sections,
            "contact": self.contact,
            "skills": self.skills,
            "source_filename": self.source_filename,
            "parsed_at": self.parsed_at.isoformat(),
        }


@dataclass
class MatchResult:
    job: Job
    overall_score: float
    section_scores: dict[str, float]
    fraud_score: float
    fraud_reasons: list[str]
    matched_keywords: list[str]
    missing_keywords: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "job": self.job.to_dict(),
            "overall_score": self.overall_score,
            "section_scores": self.section_scores,
            "fraud_score": self.fraud_score,
            "fraud_reasons": self.fraud_reasons,
            "matched_keywords": self.matched_keywords,
            "missing_keywords": self.missing_keywords,
        }
