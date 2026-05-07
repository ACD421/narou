from datetime import datetime, timedelta, timezone

from narou.fraud import extract_batch, score_heuristics, score_jobs
from narou.schema import Job


def _job(title, desc, days=5, company="acme", job_id=None):
    jid = job_id or title.replace(" ", "_") + str(days)
    return Job(
        job_id=jid,
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


CLEAN_DESC = (
    "We are hiring a Senior Security Engineer to build detection pipelines on AWS. "
    "You will work with Python, Splunk, Kafka, and the SIEM team. Required: 5+ years "
    "of experience with SOC or incident response. Salary range $170,000 - $220,000."
)

BUZZ_DESC = (
    "We're a passionate, fast-paced unicorn looking for a rockstar ninja guru wizard. "
    "You will be a world-class industry-leading cutting-edge game-changing synergy hustler."
)


def test_extract_batch_produces_features_per_job():
    jobs = [
        _job("Security Engineer", CLEAN_DESC),
        _job("Sales Rep", BUZZ_DESC),
    ]
    feats = extract_batch(jobs)
    assert len(feats) == 2
    assert feats[0].word_count > 0
    assert feats[0].has_salary == 1
    assert feats[0].tech_density > 0
    assert feats[1].buzzword_ratio > feats[0].buzzword_ratio


def test_heuristics_flags_stale_postings():
    f = extract_batch([_job("X", CLEAN_DESC, days=200)])[0]
    h = score_heuristics(f)
    assert h.score >= 0.25
    assert any("stale" in r for r in h.reasons)


def test_heuristics_flags_short_description():
    short = _job("Vague Role", "We need someone great.")
    f = extract_batch([short])[0]
    h = score_heuristics(f)
    assert h.score >= 0.25
    assert any("short" in r for r in h.reasons)


def test_heuristics_flags_duplicate_descriptions():
    dup = CLEAN_DESC
    jobs = [
        _job("Role A", dup, job_id="a"),
        _job("Role B", dup, job_id="b"),
    ]
    reports = score_jobs(jobs, classifier=None)
    flagged = [r for r in reports if r.flagged or r.score > 0.30]
    assert flagged, "duplicate description should raise fraud score"


def test_score_jobs_returns_report_per_job():
    jobs = [
        _job("Senior Eng", CLEAN_DESC),
        _job("Mystery", BUZZ_DESC),
    ]
    reports = score_jobs(jobs, classifier=None)
    assert len(reports) == 2
    clean, sus = reports
    assert sus.score > clean.score
