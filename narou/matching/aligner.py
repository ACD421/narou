from __future__ import annotations

from dataclasses import dataclass, field

from ..schema import Job, Resume
from ..utils import bullet_split, sentence_split
from .sgm import SGMEngine, get_engine, stem_overlap


@dataclass
class SectionScores:
    overall: float = 0.0
    title: float = 0.0
    summary: float = 0.0
    skills: float = 0.0
    experience: float = 0.0
    education: float = 0.0
    lexical: float = 0.0
    matched_keywords: list[str] = field(default_factory=list)
    missing_keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "overall": self.overall,
            "title": self.title,
            "summary": self.summary,
            "skills": self.skills,
            "experience": self.experience,
            "education": self.education,
            "lexical": self.lexical,
            "matched_keywords": self.matched_keywords,
            "missing_keywords": self.missing_keywords,
        }


SECTION_WEIGHTS = {
    "title": 0.30,
    "skills": 0.25,
    "lexical": 0.20,
    "summary": 0.15,
    "experience": 0.10,
}


@dataclass
class ResumeQueryCache:
    """Precomputed resume artifacts reused across all jobs in one rank query."""
    identity_text: str
    summary_text: str
    experience_text: str
    skills_text: str
    identity_doc: dict | None = None
    summary_doc: dict | None = None
    experience_doc: dict | None = None
    skills_doc: dict | None = None


def build_resume_cache(resume: Resume, engine: SGMEngine) -> ResumeQueryCache:
    identity = _resume_identity_text(resume)
    summary = (resume.sections.get("summary", "") or "")[:1500]
    experience = (resume.sections.get("experience", "") or "")[:1500]
    skills_text = (resume.sections.get("skills", "") or ", ".join(resume.skills))[:1500]
    cache = ResumeQueryCache(
        identity_text=identity,
        summary_text=summary,
        experience_text=experience,
        skills_text=skills_text,
    )
    if identity:
        cache.identity_doc = engine.precompute_doc(identity)
    if summary:
        cache.summary_doc = engine.precompute_doc(summary)
    if experience:
        cache.experience_doc = engine.precompute_doc(experience)
    if skills_text:
        cache.skills_doc = engine.precompute_doc(skills_text)
    return cache


def _score_text_against_job(engine: SGMEngine, text: str, job_text: str) -> float:
    if not text or not job_text:
        return 0.0
    text = text[:1500]
    job_text = job_text[:2500]
    return engine.similarity(text, job_text)


def _score_doc_against_job(engine: SGMEngine, doc, job_doc) -> float:
    if doc is None or job_doc is None:
        return 0.0
    return engine.similarity_pre(doc, job_doc)


def _skill_match_ratio(resume_skills: list[str], job_text: str) -> tuple[float, list[str], list[str]]:
    if not resume_skills or not job_text:
        return 0.0, [], []
    job_lower = job_text.lower()
    matched = []
    missing = []
    for skill in resume_skills:
        if not skill:
            continue
        key = skill.lower().strip()
        if len(key) < 2:
            continue
        if key in job_lower:
            matched.append(skill)
        else:
            missing.append(skill)
    total = len(resume_skills)
    if total == 0:
        return 0.0, [], []
    return len(matched) / total, matched, missing


def _resume_identity_text(resume: Resume) -> str:
    parts = []
    summary = resume.sections.get("summary", "")
    if summary:
        parts.append(summary[:400])
    if resume.skills:
        parts.append(", ".join(resume.skills[:15]))
    return "\n".join(parts)[:800]


def score_resume_vs_job(
    resume: Resume,
    job: Job,
    engine: SGMEngine | None = None,
    resume_cache: ResumeQueryCache | None = None,
    stage1_lexical: float | None = None,
    job_sgm: dict | None = None,
) -> SectionScores:
    engine = engine or get_engine()
    job_text = f"{job.title}\n{job.department or ''}\n{job.description}"[:2500]
    job_header = f"{job.title} {job.department or ''}".strip()

    # Precompute job-side once per query (or use cached version if supplied).
    if job_sgm is not None:
        job_doc = job_sgm.get("body")
        job_header_doc = job_sgm.get("header")
    else:
        job_doc = engine.precompute_doc(job_text) if job_text else None
        job_header_doc = engine.precompute_doc(job_header) if job_header else None

    rc = resume_cache
    if rc is None:
        rc = build_resume_cache(resume, engine)

    scores = SectionScores()

    title_sgm = (
        engine.similarity_pre(rc.identity_doc, job_header_doc)
        if rc.identity_doc is not None and job_header_doc is not None
        else 0.0
    )
    title_fast = SGMEngine.fast_similarity(rc.identity_text, job_header)
    scores.title = max(title_sgm, title_fast * 1.2)

    scores.summary = _score_doc_against_job(engine, rc.summary_doc, job_doc)
    scores.experience = _score_doc_against_job(engine, rc.experience_doc, job_doc)

    skills_sgm = _score_doc_against_job(engine, rc.skills_doc, job_doc)
    skills_ratio, matched, missing = _skill_match_ratio(resume.skills, job_text)
    scores.skills = 0.5 * skills_sgm + 0.5 * skills_ratio
    scores.matched_keywords = matched
    scores.missing_keywords = missing

    if stage1_lexical is not None:
        scores.lexical = stage1_lexical
    else:
        stage1_text = resume.sections.get("summary", "")
        if resume.skills:
            stage1_text += " " + ", ".join(resume.skills[:20])
        if resume.sections.get("experience"):
            stage1_text += " " + resume.sections["experience"][:800]
        scores.lexical = SGMEngine.fast_similarity(stage1_text[:2500], job_text[:2500])

    weighted = 0.0
    weight_sum = 0.0
    for key, w in SECTION_WEIGHTS.items():
        val = getattr(scores, key)
        if val > 0:
            weighted += w * val
            weight_sum += w
    if weight_sum > 0:
        scores.overall = weighted / weight_sum
    else:
        fallback_text = resume.summary_text()[:2000]
        scores.overall = _score_text_against_job(engine, fallback_text, job_text)
    return scores


def align_sentences(
    resume_text: str,
    job_text: str,
    engine: SGMEngine | None = None,
    top_k: int = 3,
) -> list[tuple[str, str, float]]:
    engine = engine or get_engine()
    r_sents = sentence_split(resume_text) + bullet_split(resume_text)
    j_sents = sentence_split(job_text) + bullet_split(job_text)
    if not r_sents or not j_sents:
        return []

    pairs: list[tuple[str, str, float]] = []
    for rs in r_sents[:20]:
        best: tuple[str, str, float] | None = None
        for js in j_sents[:30]:
            if stem_overlap(rs, js) < 0.05:
                continue
            sim = engine.similarity(rs, js)
            if best is None or sim > best[2]:
                best = (rs, js, sim)
        if best is not None:
            pairs.append(best)

    pairs.sort(key=lambda x: -x[2])
    return pairs[:top_k]
