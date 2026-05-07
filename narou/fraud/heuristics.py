from __future__ import annotations

from dataclasses import dataclass, field

from .features import FraudFeatures


@dataclass
class HeuristicResult:
    score: float
    reasons: list[str] = field(default_factory=list)

    def clamp(self) -> "HeuristicResult":
        self.score = max(0.0, min(1.0, self.score))
        return self


def score_heuristics(f: FraudFeatures) -> HeuristicResult:
    result = HeuristicResult(score=0.0)

    if f.days_active > 120:
        result.score += 0.30
        result.reasons.append(f"open {int(f.days_active)} days (very stale)")
    elif f.days_active > 60:
        result.score += 0.15
        result.reasons.append(f"open {int(f.days_active)} days (stale)")

    if f.repost_count >= 3:
        result.score += 0.25
        result.reasons.append(f"title reposted {f.repost_count}x by same company")
    elif f.repost_count >= 2:
        result.score += 0.12
        result.reasons.append(f"title reposted {f.repost_count}x by same company")

    if f.dup_similarity >= 1.0:
        result.score += 0.35
        result.reasons.append("exact description duplicate of another listing")

    if f.word_count < 80:
        result.score += 0.25
        result.reasons.append(f"unusually short description ({f.word_count} words)")
    elif f.word_count > 1200 and f.specificity < 0.35:
        result.score += 0.10
        result.reasons.append("long but low-specificity description (padding)")

    if f.buzzword_ratio > 0.25:
        result.score += 0.15
        result.reasons.append("heavy buzzword language")
    elif f.buzzword_ratio > 0.10:
        result.score += 0.07
        result.reasons.append("mild buzzword language")

    if f.tech_density == 0 and f.word_count > 200:
        result.score += 0.08
        result.reasons.append("no concrete technical keywords")

    if f.caps_ratio > 0.18:
        result.score += 0.10
        result.reasons.append("excessive capitalization")

    if f.exclamation_density > 0.6:
        result.score += 0.06
        result.reasons.append("excessive exclamation marks")

    if not f.has_years and f.word_count > 250 and f.word_count < 400:
        result.score += 0.05
        result.reasons.append("vague experience requirements")

    if f.company_title_repeat_rate > 0.30:
        result.score += 0.12
        result.reasons.append(
            f"{int(f.company_title_repeat_rate * 100)}% of this company's listings have duplicate titles"
        )

    if f.specificity and f.specificity < 0.25 and f.word_count > 200:
        result.score += 0.08
        result.reasons.append("generic boilerplate language")

    if f.company_posting_velocity > 1.5 and f.repost_count > 0:
        result.score += 0.05
        result.reasons.append("high posting velocity with repeated titles")

    return result.clamp()
