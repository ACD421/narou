from datetime import datetime, timedelta, timezone

from narou.fraud import score_jobs
from narou.grading import grade_companies
from narou.schema import Job


def _job(company, title, days, desc):
    return Job(
        job_id=f"{company}-{title}-{days}",
        source="greenhouse",
        company=company,
        title=title,
        location="Remote",
        department=None,
        description=desc,
        url="",
        posted_at=datetime.now(timezone.utc) - timedelta(days=days),
        updated_at=None,
    )


def test_clean_company_gets_high_grade():
    desc = (
        "We are hiring a Senior Engineer to build distributed systems in Python and Go. "
        "Required: 5+ years experience with AWS, Kubernetes, and PostgreSQL. "
        "Salary $180,000 - $230,000."
    )
    jobs = [
        _job("clean-co", f"Engineer {i}", days=10, desc=desc.replace("Python", f"lang{i}"))
        for i in range(5)
    ]
    reports = score_jobs(jobs, classifier=None)
    grades = grade_companies(jobs, reports)
    assert len(grades) == 1
    assert grades[0].letter in ("A", "B")


def test_stale_company_gets_low_grade():
    desc = (
        "Seeking a passionate unicorn rockstar ninja guru wizard for a dynamic environment. "
        "World-class opportunity in a fast-paced industry-leading cutting-edge synergy team."
    )
    jobs = [
        _job("shady-co", "Mystery Role", days=180, desc=desc),
        _job("shady-co", "Mystery Role", days=200, desc=desc),
        _job("shady-co", "Mystery Role", days=250, desc=desc),
        _job("shady-co", "Another Vague", days=220, desc=desc),
    ]
    reports = score_jobs(jobs, classifier=None)
    grades = grade_companies(jobs, reports)
    assert len(grades) == 1
    assert grades[0].letter in ("D", "F")
    assert grades[0].flagged_rate > 0
