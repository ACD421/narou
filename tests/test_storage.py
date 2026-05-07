from datetime import datetime, timezone
from pathlib import Path

from narou.schema import Job
from narou.storage import Database


def _sample_job(i=1, company="acme"):
    return Job(
        job_id=str(i),
        source="greenhouse",
        company=company,
        title=f"Engineer {i}",
        location="Remote",
        department="Eng",
        description=f"Build cool stuff {i}",
        url=f"https://example.com/{i}",
        posted_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def test_upsert_and_list(tmp_path: Path):
    db = Database(tmp_path / "jobs.sqlite")
    assert db.upsert_jobs([_sample_job(1), _sample_job(2)]) == 2
    jobs = db.list_jobs()
    assert len(jobs) == 2


def test_upsert_is_idempotent(tmp_path: Path):
    db = Database(tmp_path / "jobs.sqlite")
    db.upsert_jobs([_sample_job(1)])
    db.upsert_jobs([_sample_job(1)])
    assert len(db.list_jobs()) == 1


def test_list_by_company(tmp_path: Path):
    db = Database(tmp_path / "jobs.sqlite")
    db.upsert_jobs([_sample_job(1, "alpha"), _sample_job(2, "beta")])
    assert len(db.list_jobs(company="alpha")) == 1
    assert len(db.list_jobs(company="beta")) == 1


def test_cache_roundtrip(tmp_path: Path):
    db = Database(tmp_path / "jobs.sqlite")
    db.cache_put("k1", "ok", {"a": 1, "b": [2, 3]})
    got = db.cache_get("k1", ttl=60)
    assert got is not None
    assert got["payload"] == {"a": 1, "b": [2, 3]}


def test_cache_expires(tmp_path: Path):
    db = Database(tmp_path / "jobs.sqlite")
    db.cache_put("k1", "ok", {"a": 1})
    assert db.cache_get("k1", ttl=0) is None


def test_runs_recorded(tmp_path: Path):
    db = Database(tmp_path / "jobs.sqlite")
    db.record_run("r1", boards_scanned=2, jobs_ingested=10, jobs_matched=5, jobs_flagged=1, elapsed_ms=100.0)
    runs = db.recent_runs()
    assert len(runs) == 1
    assert runs[0]["boards_scanned"] == 2


def test_stats_reflects_data(tmp_path: Path):
    db = Database(tmp_path / "jobs.sqlite")
    db.upsert_jobs([_sample_job(1, "alpha"), _sample_job(2, "alpha"), _sample_job(3, "beta")])
    s = db.stats()
    assert s["total_jobs"] == 3
    assert s["distinct_companies"] == 2
    assert s["by_source"] == {"greenhouse": 3}
