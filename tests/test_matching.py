from datetime import datetime, timezone

from narou.matching import get_engine, rank_jobs, score_resume_vs_job
from narou.matching.sgm import SGMEngine
from narou.schema import Job, Resume


def _resume(summary: str, skills: list[str]) -> Resume:
    return Resume(
        raw_text=summary + "\nSKILLS\n" + ", ".join(skills),
        sections={"summary": summary, "skills": ", ".join(skills)},
        contact={"email": "x@example.com"},
        skills=skills,
        source_filename="r.pdf",
    )


def _job(title: str, desc: str, company: str = "acme") -> Job:
    return Job(
        job_id=title.replace(" ", "_"),
        source="greenhouse",
        company=company,
        title=title,
        location="Remote",
        department=None,
        description=desc,
        url="",
        posted_at=datetime.now(timezone.utc),
        updated_at=None,
    )


def test_engine_loads_and_reports_size():
    eng = get_engine()
    assert eng.size_mb > 0
    assert len(eng.feat_names) == 20


def test_fast_similarity_separates_related_from_unrelated():
    a = "security engineer protecting enterprise cloud infrastructure"
    related = "detection engineer for aws workloads in cloud environments"
    unrelated = "product marketing manager for consumer mobile apps"
    rel = SGMEngine.fast_similarity(a, related)
    unrel = SGMEngine.fast_similarity(a, unrelated)
    assert rel > unrel
    assert rel > 0.10


def test_sgm_similarity_is_bounded():
    eng = get_engine()
    score = eng.similarity("hello world", "hello world")
    assert 0.0 <= score <= 1.0


def test_score_resume_vs_job_prefers_relevant_roles():
    resume = _resume(
        "Security engineer with 8 years building detection and response systems.",
        ["Python", "AWS", "SIEM", "Splunk"],
    )
    good = _job(
        "Senior Security Engineer",
        "We are hiring a security engineer to build detection pipelines on AWS using Splunk and Python.",
    )
    bad = _job(
        "Product Marketing Manager",
        "Lead product marketing for consumer mobile apps, drive campaigns and brand.",
    )
    good_score = score_resume_vs_job(resume, good).overall
    bad_score = score_resume_vs_job(resume, bad).overall
    assert good_score > bad_score


def test_rank_jobs_returns_sorted():
    resume = _resume(
        "Senior Python backend engineer.",
        ["Python", "FastAPI", "Postgres"],
    )
    jobs = [
        _job("Senior Backend Engineer (Python)", "Build APIs in FastAPI with Postgres."),
        _job("Frontend Designer", "Figma, CSS, brand design"),
        _job("Data Scientist", "Statistical modeling, experimentation"),
    ]
    ranked = rank_jobs(resume, jobs, top_n=3)
    assert len(ranked) == 3
    assert ranked[0].job.title.startswith("Senior Backend")
    assert ranked[0].overall >= ranked[-1].overall
