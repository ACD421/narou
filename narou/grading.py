from __future__ import annotations

from dataclasses import dataclass, field

from .fraud.scorer import FraudReport
from .schema import Job


@dataclass
class CompanyGrade:
    company: str
    source: str
    posting_volume: int
    flagged_count: int
    flagged_rate: float
    median_days_active: float
    repost_rate: float
    avg_fraud_score: float
    letter: str
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "company": self.company,
            "source": self.source,
            "posting_volume": self.posting_volume,
            "flagged_count": self.flagged_count,
            "flagged_rate": self.flagged_rate,
            "median_days_active": self.median_days_active,
            "repost_rate": self.repost_rate,
            "avg_fraud_score": self.avg_fraud_score,
            "letter": self.letter,
            "reasons": self.reasons,
        }


def _letter_from_rate(rate: float) -> str:
    if rate < 0.05:
        return "A"
    if rate < 0.12:
        return "B"
    if rate < 0.25:
        return "C"
    if rate < 0.40:
        return "D"
    return "F"


def grade_companies(
    jobs: list[Job],
    reports: list[FraudReport],
) -> list[CompanyGrade]:
    by_uid = {rp.job_uid: rp for rp in reports}
    by_company: dict[tuple[str, str], list[Job]] = {}
    for j in jobs:
        by_company.setdefault((j.company, j.source), []).append(j)

    grades: list[CompanyGrade] = []
    for (company, source), cjobs in by_company.items():
        flagged = [j for j in cjobs if by_uid.get(j.uid) and by_uid[j.uid].flagged]
        scores = [by_uid[j.uid].score for j in cjobs if j.uid in by_uid]
        days = sorted([j.days_active for j in cjobs if j.days_active is not None])
        median_days = days[len(days) // 2] if days else 0.0

        title_counts: dict[str, int] = {}
        for j in cjobs:
            t = j.title.strip().lower()
            title_counts[t] = title_counts.get(t, 0) + 1
        repeats = sum(1 for c in title_counts.values() if c > 1)
        repost_rate = repeats / len(cjobs) if cjobs else 0.0

        flagged_rate = len(flagged) / len(cjobs) if cjobs else 0.0
        avg_score = sum(scores) / len(scores) if scores else 0.0
        letter = _letter_from_rate(flagged_rate)

        reasons: list[str] = []
        if flagged_rate >= 0.12:
            reasons.append(f"{int(flagged_rate * 100)}% of listings flagged as ghost-risk")
        if median_days > 60:
            reasons.append(f"median posting age {int(median_days)} days")
        if repost_rate > 0.20:
            reasons.append(f"{int(repost_rate * 100)}% of listings have duplicate titles")
        if not reasons:
            reasons.append("No significant ghost-job indicators")

        grades.append(CompanyGrade(
            company=company,
            source=source,
            posting_volume=len(cjobs),
            flagged_count=len(flagged),
            flagged_rate=flagged_rate,
            median_days_active=median_days,
            repost_rate=repost_rate,
            avg_fraud_score=avg_score,
            letter=letter,
            reasons=reasons,
        ))

    grades.sort(key=lambda g: (g.letter, -g.posting_volume))
    return grades
