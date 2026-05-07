from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from ..fraud.features import TECH_KEYWORDS
from ..matching.ranker import RankedMatch
from ..matching.sgm import _cw, _stem
from ..schema import Resume


STOPSKILL = {
    "team", "work", "role", "experience", "ability", "knowledge", "years",
    "english", "strong", "excellent", "good", "ideal", "candidate", "must",
    "required", "preferred", "plus", "responsibilities", "qualifications",
    "minimum", "maximum", "will", "can", "may", "help", "build", "make",
    "support", "develop", "manage", "design", "lead", "create", "other",
    "employees", "people", "company", "customer", "customers", "user",
    "users", "product", "products", "business", "project", "projects",
}

_BIGRAM_RE = re.compile(r"\b([a-z][a-z\-]{2,})\s+([a-z][a-z\-]{2,})\b")
_ACRONYM_RE = re.compile(r"\b([A-Z]{3,6})\b")


def _word_search(keyword: str) -> re.Pattern:
    escaped = re.escape(keyword)
    return re.compile(rf"(?<![a-zA-Z]){escaped}(?![a-zA-Z])", re.IGNORECASE)


@dataclass
class SuggestionReport:
    headline: str
    summary: str
    top_missing_keywords: list[tuple[str, int]] = field(default_factory=list)
    strong_keywords: list[tuple[str, int]] = field(default_factory=list)
    section_advice: dict[str, str] = field(default_factory=dict)
    rephrase_hints: list[str] = field(default_factory=list)
    common_themes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "headline": self.headline,
            "summary": self.summary,
            "top_missing_keywords": self.top_missing_keywords,
            "strong_keywords": self.strong_keywords,
            "section_advice": self.section_advice,
            "rephrase_hints": self.rephrase_hints,
            "common_themes": self.common_themes,
        }


def _resume_vocab(resume: Resume) -> set[str]:
    blob_parts = [resume.raw_text]
    blob_parts.extend(resume.skills)
    blob = " ".join(blob_parts).lower()
    stems = {_stem(w) for w in _cw(blob)}
    # Keep full skill phrases so multi-word matches like "incident response" work
    stems.update(w.lower() for w in resume.skills if w)
    # Also keep the raw_text as a searchable string for substring fallback
    stems.add("__raw__:" + blob)
    return stems


_KW_PATTERNS = {
    kw: _word_search(kw) for kw in TECH_KEYWORDS if len(kw) >= 3
}

_ACRONYM_STOP = {"AND", "FOR", "THE", "BUT", "NOT", "YOU", "HIS", "HER", "OUR"}


def _job_keyword_counter(matches: Iterable[RankedMatch]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for m in matches:
        text = f"{m.job.title}\n{m.job.description or ''}"
        for kw, pattern in _KW_PATTERNS.items():
            if pattern.search(text):
                counter[kw] += 1
        for acr in _ACRONYM_RE.findall(m.job.description or ""):
            if acr.upper() in _ACRONYM_STOP:
                continue
            counter[acr.lower()] += 1
    return counter


def _section_advice(resume: Resume, matches: list[RankedMatch]) -> dict[str, str]:
    advice: dict[str, str] = {}
    if not matches:
        return advice

    avg_title = sum(m.scores.title for m in matches) / len(matches)
    avg_skills = sum(m.scores.skills for m in matches) / len(matches)
    avg_summary = sum(m.scores.summary for m in matches) / len(matches)
    avg_experience = sum(m.scores.experience for m in matches) / len(matches)

    if avg_title < 0.35:
        advice["summary"] = (
            "Your resume's professional title and opening summary don't clearly match the "
            "language of your top target roles. Rewrite your headline and first sentence using "
            "the exact role titles you're targeting."
        )
    if avg_skills < 0.30 and resume.skills:
        advice["skills"] = (
            "Your skills section uses specialty certifications that don't literally appear in "
            "most job descriptions. Add a short list of concrete tools and technologies "
            "(languages, frameworks, platforms) alongside your specialized competencies."
        )
    if not resume.skills and "skills" not in resume.sections:
        advice["skills"] = (
            "Your resume has no dedicated Skills section. Add one listing the tools and "
            "technologies from your experience. ATS parsers and keyword-based filters look for "
            "this section first."
        )
    if avg_summary < 0.30 and "summary" not in resume.sections:
        advice["summary"] = (
            "You don't have a dedicated Summary or Profile section. Write a 2-3 sentence opener "
            "describing your target role, years of experience, and top 3 skills."
        )
    if avg_experience < 0.30:
        advice["experience"] = (
            "Your experience bullets are not matching the language of target job descriptions. "
            "Rewrite bullets in the form 'Delivered X using Y, resulting in Z', mirroring the "
            "verbs and technologies used in the listings."
        )
    return advice


def _rephrase_hints(
    resume: Resume,
    missing: list[tuple[str, int]],
    matches: list[RankedMatch],
) -> list[str]:
    hints: list[str] = []
    if not missing or not matches:
        return hints

    top_titles = [m.job.title for m in matches[:3]]
    top_kw = [kw for kw, _ in missing[:5]]
    if top_kw:
        kw_list = ", ".join(top_kw)
        hints.append(
            f"Add a single bullet to your experience section that names these "
            f"technologies directly: {kw_list}. Job descriptions at your top matches "
            f"({', '.join(top_titles[:2])}) all mention them."
        )
    if resume.sections.get("summary"):
        hints.append(
            "Rework your opening summary to name the exact job title you're applying to. "
            "ATS keyword filters weight title matches heavily."
        )
    if len(resume.skills) < 8:
        hints.append(
            "Your skills section is short. Add the 5-10 most common technologies from your "
            "top matched jobs if you have genuine experience with them."
        )
    missing_acronyms = [k for k, _ in missing if k.isupper() or len(k) <= 5]
    if missing_acronyms[:3]:
        hints.append(
            f"Consider spelling out or adding these acronyms if you have relevant experience: "
            f"{', '.join(missing_acronyms[:3]).upper()}."
        )
    return hints


def _common_themes(matches: list[RankedMatch]) -> list[str]:
    if not matches:
        return []
    bigram_counter: Counter[str] = Counter()
    for m in matches:
        text = f"{m.job.title} {m.job.description}".lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        for a, b in _BIGRAM_RE.findall(text):
            if a in STOPSKILL or b in STOPSKILL:
                continue
            if len(a) < 3 or len(b) < 3:
                continue
            bigram_counter[f"{a} {b}"] += 1
    min_count = max(2, len(matches) // 3)
    themes = [bg for bg, c in bigram_counter.most_common(25) if c >= min_count]
    return themes[:8]


def generate_suggestions(
    resume: Resume,
    matches: list[RankedMatch],
    top_k_for_gap: int = 10,
) -> SuggestionReport:
    if not matches:
        return SuggestionReport(
            headline="No matches yet",
            summary="Run an analysis against at least one board to get tailored suggestions.",
        )

    top_matches = matches[:top_k_for_gap]
    kw_counter = _job_keyword_counter(top_matches)
    resume_vocab = _resume_vocab(resume)

    # Extract the raw text blob for substring fallback matching
    raw_blob = ""
    for item in resume_vocab:
        if item.startswith("__raw__:"):
            raw_blob = item[8:]
            break
    resume_vocab_clean = {v for v in resume_vocab if not v.startswith("__raw__:")}

    missing: list[tuple[str, int]] = []
    strong: list[tuple[str, int]] = []
    for kw, count in kw_counter.most_common():
        kw_stem = _stem(kw)
        kw_low = kw.lower()
        in_resume = (
            kw_low in resume_vocab_clean
            or kw_stem in resume_vocab_clean
            or any(kw_low in s.lower() for s in resume.skills)
            or kw_low in raw_blob
        )
        # Multi-word keywords like "iso 27001": check if all words appear
        # even when separated by other tokens (handles "ISO/IEC 27001")
        if not in_resume and " " in kw_low:
            parts = kw_low.split()
            if all(p in raw_blob for p in parts):
                in_resume = True
        if in_resume:
            strong.append((kw, count))
        else:
            missing.append((kw, count))

    missing = [(k, c) for k, c in missing if c >= max(2, len(top_matches) // 4)][:12]
    strong = strong[:10]
    section_advice = _section_advice(resume, top_matches)
    rephrase_hints = _rephrase_hints(resume, missing, top_matches)
    common_themes = _common_themes(top_matches)

    top1 = top_matches[0]
    headline = (
        f"Your strongest match is {top1.job.title} at {top1.job.company.title()} "
        f"({top1.overall*100:.0f}% overall)."
    )
    parts = []
    if strong:
        parts.append(
            f"You already match on {len(strong)} keywords including {', '.join(k for k,_ in strong[:4])}."
        )
    if missing:
        parts.append(
            f"{len(missing)} keywords appear in your top matches but not your resume: "
            f"{', '.join(k for k,_ in missing[:4])}."
        )
    if not parts:
        parts.append("Your resume aligns well with your top matches. Focus on tailoring cover letters.")
    summary = " ".join(parts)

    return SuggestionReport(
        headline=headline,
        summary=summary,
        top_missing_keywords=missing,
        strong_keywords=strong,
        section_advice=section_advice,
        rephrase_hints=rephrase_hints,
        common_themes=common_themes,
    )
