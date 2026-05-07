from datetime import datetime, timedelta, timezone

from narou.schema import Job, Resume


def _job(**overrides):
    defaults = dict(
        job_id="1",
        source="greenhouse",
        company="acme",
        title="Staff Engineer",
        location="Remote",
        department="Eng",
        description="Build cool stuff in Python.",
        url="https://example.com/1",
        posted_at=datetime.now(timezone.utc) - timedelta(days=5),
        updated_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    return Job(**defaults)


def test_job_uid_combines_source_and_id():
    j = _job()
    assert j.uid == "greenhouse:1"


def test_job_description_hash_is_deterministic():
    a = _job()
    b = _job()
    assert a.description_hash == b.description_hash
    assert len(a.description_hash) == 16


def test_job_description_hash_ignores_whitespace_and_case():
    a = _job(description="Build COOL stuff in Python.")
    b = _job(description="build  cool stuff\tin   python.")
    assert a.description_hash == b.description_hash


def test_job_days_active_positive_and_bounded():
    j = _job(posted_at=datetime.now(timezone.utc) - timedelta(days=12))
    assert j.days_active == 12


def test_job_days_active_is_none_when_unposted():
    j = _job(posted_at=None)
    assert j.days_active is None


def test_job_to_dict_serializes_timestamps():
    j = _job()
    d = j.to_dict()
    assert d["uid"] == "greenhouse:1"
    assert isinstance(d["posted_at"], str)
    assert isinstance(d["fetched_at"], str)
    assert d["days_active"] is not None


def test_resume_summary_text_prefers_sections():
    r = Resume(
        raw_text="raw body",
        sections={"summary": "SUM", "experience": "EXP", "skills": "SK"},
        contact={},
        skills=[],
        source_filename="r.pdf",
    )
    text = r.summary_text()
    assert "SUM" in text and "EXP" in text and "SK" in text


def test_resume_summary_text_falls_back_to_raw():
    r = Resume(raw_text="fallback", sections={}, contact={}, skills=[], source_filename="r.pdf")
    assert r.summary_text() == "fallback"
