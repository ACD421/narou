from datetime import datetime, timezone

from narou.matching.aligner import SectionScores
from narou.matching.ranker import RankedMatch
from narou.schema import Job, Resume
from narou.suggestions import generate_suggestions


def _ranked(title: str, desc: str, overall: float = 0.6) -> RankedMatch:
    job = Job(
        job_id=title.replace(" ", "_"),
        source="greenhouse",
        company="acme",
        title=title,
        location="Remote",
        department=None,
        description=desc,
        url="",
        posted_at=datetime.now(timezone.utc),
        updated_at=None,
    )
    scores = SectionScores(
        overall=overall, title=0.6, skills=0.5, summary=0.4, experience=0.4, lexical=0.4
    )
    return RankedMatch(job=job, scores=scores, stage1_score=0.3)


def test_generate_suggestions_identifies_missing_keywords():
    resume = Resume(
        raw_text="Python engineer",
        sections={"summary": "Python engineer.", "skills": "Python"},
        contact={},
        skills=["Python"],
        source_filename="r.pdf",
    )
    matches = [
        _ranked("Backend Engineer", "Build services in Python with Kubernetes and PostgreSQL. Use AWS."),
        _ranked("Platform Engineer", "Kubernetes and AWS experience required. Terraform a plus."),
        _ranked("Senior Backend", "Postgres, AWS, Kubernetes. Python for scripting."),
    ]
    sug = generate_suggestions(resume, matches)
    missing_kws = [k for k, _ in sug.top_missing_keywords]
    assert "kubernetes" in missing_kws or "aws" in missing_kws or "postgres" in missing_kws


def test_generate_suggestions_identifies_strong_keywords():
    resume = Resume(
        raw_text="Python AWS Kubernetes",
        sections={"summary": "Python AWS Kubernetes engineer.", "skills": "Python, AWS, Kubernetes"},
        contact={},
        skills=["Python", "AWS", "Kubernetes"],
        source_filename="r.pdf",
    )
    matches = [
        _ranked("Backend Engineer", "Python AWS Kubernetes required."),
        _ranked("Platform Engineer", "Kubernetes and AWS experience. Python scripting."),
    ]
    sug = generate_suggestions(resume, matches)
    strong_kws = [k for k, _ in sug.strong_keywords]
    assert any(k in strong_kws for k in ("python", "aws", "kubernetes"))


def test_generate_suggestions_with_no_matches_returns_empty():
    resume = Resume(
        raw_text="x", sections={}, contact={}, skills=[], source_filename="r.pdf"
    )
    sug = generate_suggestions(resume, [])
    assert sug.headline
    assert not sug.top_missing_keywords
