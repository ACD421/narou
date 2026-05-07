from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from ..schema import Job
from ..utils import parse_iso_datetime


SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    uid TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    job_id TEXT NOT NULL,
    company TEXT NOT NULL,
    title TEXT NOT NULL,
    location TEXT,
    department TEXT,
    description TEXT,
    url TEXT,
    posted_at TEXT,
    updated_at TEXT,
    fetched_at TEXT NOT NULL,
    description_hash TEXT,
    raw_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_company ON jobs(company);
CREATE INDEX IF NOT EXISTS idx_jobs_source ON jobs(source);
CREATE INDEX IF NOT EXISTS idx_jobs_hash ON jobs(description_hash);
CREATE INDEX IF NOT EXISTS idx_jobs_fetched ON jobs(fetched_at);

CREATE TABLE IF NOT EXISTS feed_cache (
    key TEXT PRIMARY KEY,
    fetched_at REAL NOT NULL,
    status TEXT NOT NULL,
    payload TEXT
);

CREATE TABLE IF NOT EXISTS analyses (
    job_uid TEXT NOT NULL,
    resume_hash TEXT NOT NULL,
    overall_score REAL,
    section_scores TEXT,
    fraud_score REAL,
    fraud_reasons TEXT,
    created_at REAL NOT NULL,
    PRIMARY KEY (job_uid, resume_hash)
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    started_at REAL NOT NULL,
    finished_at REAL,
    boards_scanned INTEGER,
    jobs_ingested INTEGER,
    jobs_matched INTEGER,
    jobs_flagged INTEGER,
    elapsed_ms REAL,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS board_health (
    source TEXT NOT NULL,
    board TEXT NOT NULL,
    last_ok_at REAL,
    last_attempt_at REAL,
    last_status TEXT,
    last_error TEXT,
    jobs_seen INTEGER DEFAULT 0,
    consecutive_failures INTEGER DEFAULT 0,
    PRIMARY KEY (source, board)
);

CREATE TABLE IF NOT EXISTS job_features (
    uid TEXT PRIMARY KEY,
    description_hash TEXT,
    computed_at REAL NOT NULL,
    features_json TEXT,
    features_blob BLOB,
    sgm_blob BLOB,
    FOREIGN KEY (uid) REFERENCES jobs(uid) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_job_features_hash ON job_features(description_hash);

CREATE TABLE IF NOT EXISTS crawl_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL
);
"""


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(
                self.path,
                detect_types=sqlite3.PARSE_DECLTYPES,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-8000")  # 8 MB
            self._local.conn = conn
        return conn

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = self._get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def upsert_jobs(self, jobs: list[Job]) -> int:
        if not jobs:
            return 0
        rows = []
        for j in jobs:
            rows.append((
                j.uid, j.source, j.job_id, j.company, j.title, j.location,
                j.department, j.description, j.url,
                j.posted_at.isoformat() if j.posted_at else None,
                j.updated_at.isoformat() if j.updated_at else None,
                j.fetched_at.isoformat(),
                j.description_hash,
                json.dumps(j.raw, default=str),
            ))
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO jobs(uid, source, job_id, company, title, location, department,
                                 description, url, posted_at, updated_at, fetched_at,
                                 description_hash, raw_json)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(uid) DO UPDATE SET
                    title=excluded.title,
                    location=excluded.location,
                    department=excluded.department,
                    description=excluded.description,
                    url=excluded.url,
                    posted_at=COALESCE(excluded.posted_at, jobs.posted_at),
                    updated_at=excluded.updated_at,
                    fetched_at=excluded.fetched_at,
                    description_hash=excluded.description_hash,
                    raw_json=excluded.raw_json
                """,
                rows,
            )
        return len(rows)

    def list_all_jobs(self, max_age_days: int | None = None) -> list[Job]:
        """Return every job in the corpus, optionally filtered by freshness."""
        sql = "SELECT * FROM jobs"
        params: list[Any] = []
        if max_age_days is not None:
            cutoff = time.time() - max_age_days * 86400
            sql += " WHERE strftime('%s', fetched_at) > ?"
            params.append(str(int(cutoff)))
        with self.connect() as conn:
            cur = conn.execute(sql, params)
            return [self._row_to_job(r) for r in cur.fetchall()]

    def list_jobs(
        self,
        company: str | None = None,
        source: str | None = None,
        limit: int = 500,
    ) -> list[Job]:
        sql = "SELECT * FROM jobs"
        clauses = []
        params: list[Any] = []
        if company:
            clauses.append("company = ?")
            params.append(company)
        if source:
            clauses.append("source = ?")
            params.append(source)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY fetched_at DESC LIMIT ?"
        params.append(limit)
        with self.connect() as conn:
            cur = conn.execute(sql, params)
            return [self._row_to_job(r) for r in cur.fetchall()]

    def company_history(self, company: str, within_days: int = 180) -> list[Job]:
        cutoff = (time.time() - within_days * 86400)
        with self.connect() as conn:
            cur = conn.execute(
                "SELECT * FROM jobs WHERE company = ? AND strftime('%s', fetched_at) > ?",
                (company, str(int(cutoff))),
            )
            return [self._row_to_job(r) for r in cur.fetchall()]

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> Job:
        raw = {}
        try:
            raw = json.loads(row["raw_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            raw = {}
        return Job(
            job_id=row["job_id"],
            source=row["source"],
            company=row["company"],
            title=row["title"],
            location=row["location"] or "",
            department=row["department"],
            description=row["description"] or "",
            url=row["url"] or "",
            posted_at=parse_iso_datetime(row["posted_at"]),
            updated_at=parse_iso_datetime(row["updated_at"]),
            fetched_at=parse_iso_datetime(row["fetched_at"]) or datetime.now(),
            description_hash=row["description_hash"] or "",
            raw=raw,
        )

    def cache_get(self, key: str, ttl: float) -> dict | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT fetched_at, status, payload FROM feed_cache WHERE key = ?", (key,)
            ).fetchone()
        if not row:
            return None
        if time.time() - row["fetched_at"] > ttl:
            return None
        try:
            return {
                "status": row["status"],
                "payload": json.loads(row["payload"]) if row["payload"] else None,
                "fetched_at": row["fetched_at"],
            }
        except json.JSONDecodeError:
            return None

    def cache_put(self, key: str, status: str, payload: Any) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO feed_cache(key, fetched_at, status, payload)
                VALUES(?,?,?,?)
                ON CONFLICT(key) DO UPDATE SET
                    fetched_at=excluded.fetched_at,
                    status=excluded.status,
                    payload=excluded.payload
                """,
                (key, time.time(), status, json.dumps(payload, default=str)),
            )

    def record_run(
        self,
        run_id: str,
        boards_scanned: int,
        jobs_ingested: int,
        jobs_matched: int,
        jobs_flagged: int,
        elapsed_ms: float,
        notes: str = "",
    ) -> None:
        now = time.time()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO runs(run_id, started_at, finished_at, boards_scanned,
                                 jobs_ingested, jobs_matched, jobs_flagged, elapsed_ms, notes)
                VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(run_id) DO UPDATE SET
                    finished_at=excluded.finished_at,
                    boards_scanned=excluded.boards_scanned,
                    jobs_ingested=excluded.jobs_ingested,
                    jobs_matched=excluded.jobs_matched,
                    jobs_flagged=excluded.jobs_flagged,
                    elapsed_ms=excluded.elapsed_ms,
                    notes=excluded.notes
                """,
                (run_id, now, now, boards_scanned, jobs_ingested, jobs_matched,
                 jobs_flagged, elapsed_ms, notes),
            )

    def recent_runs(self, limit: int = 20) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def upsert_board_health(
        self,
        source: str,
        board: str,
        ok: bool,
        status: str,
        error: str = "",
        jobs_seen: int = 0,
    ) -> None:
        now = time.time()
        with self.connect() as conn:
            row = conn.execute(
                "SELECT consecutive_failures FROM board_health WHERE source=? AND board=?",
                (source, board),
            ).fetchone()
            prev_fails = row["consecutive_failures"] if row else 0
            if ok:
                conn.execute(
                    """
                    INSERT INTO board_health(source, board, last_ok_at, last_attempt_at,
                        last_status, last_error, jobs_seen, consecutive_failures)
                    VALUES(?,?,?,?,?,?,?,0)
                    ON CONFLICT(source, board) DO UPDATE SET
                        last_ok_at=excluded.last_ok_at,
                        last_attempt_at=excluded.last_attempt_at,
                        last_status=excluded.last_status,
                        last_error='',
                        jobs_seen=excluded.jobs_seen,
                        consecutive_failures=0
                    """,
                    (source, board, now, now, status, "", jobs_seen),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO board_health(source, board, last_ok_at, last_attempt_at,
                        last_status, last_error, jobs_seen, consecutive_failures)
                    VALUES(?,?,NULL,?,?,?,0,?)
                    ON CONFLICT(source, board) DO UPDATE SET
                        last_attempt_at=excluded.last_attempt_at,
                        last_status=excluded.last_status,
                        last_error=excluded.last_error,
                        consecutive_failures=board_health.consecutive_failures + 1
                    """,
                    (source, board, now, status, error[:500], prev_fails + 1),
                )

    def healthy_boards(self, source: str | None = None) -> list[dict]:
        with self.connect() as conn:
            sql = "SELECT source, board, last_ok_at, jobs_seen, consecutive_failures FROM board_health"
            params: list[Any] = []
            clauses = ["consecutive_failures < 5"]
            if source:
                clauses.append("source = ?")
                params.append(source)
            sql += " WHERE " + " AND ".join(clauses) + " ORDER BY last_ok_at DESC NULLS LAST"
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def crawl_meta_get(self, key: str) -> str | None:
        with self.connect() as conn:
            row = conn.execute("SELECT value FROM crawl_meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def crawl_meta_set(self, key: str, value: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO crawl_meta(key, value, updated_at) VALUES(?,?,?)
                   ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at""",
                (key, value, time.time()),
            )

    def get_job_features(self, uids: list[str]) -> dict[str, dict]:
        if not uids:
            return {}
        out: dict[str, dict] = {}
        with self.connect() as conn:
            # Chunk to avoid SQLite param limit
            CHUNK = 500
            for i in range(0, len(uids), CHUNK):
                chunk = uids[i:i + CHUNK]
                placeholders = ",".join("?" * len(chunk))
                rows = conn.execute(
                    f"SELECT uid, features_json FROM job_features WHERE uid IN ({placeholders})",
                    chunk,
                ).fetchall()
                for r in rows:
                    try:
                        out[r["uid"]] = json.loads(r["features_json"])
                    except json.JSONDecodeError:
                        continue
        return out

    def upsert_job_features(self, items: list[tuple[str, str, dict]]) -> int:
        if not items:
            return 0
        now = time.time()
        rows = [(uid, dhash, now, json.dumps(feats, default=str)) for uid, dhash, feats in items]
        with self.connect() as conn:
            conn.executemany(
                """INSERT INTO job_features(uid, description_hash, computed_at, features_json)
                   VALUES(?,?,?,?)
                   ON CONFLICT(uid) DO UPDATE SET
                       description_hash=excluded.description_hash,
                       computed_at=excluded.computed_at,
                       features_json=excluded.features_json""",
                rows,
            )
        return len(rows)

    def list_all_job_uids_and_hashes(self) -> list[tuple[str, str]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT uid, description_hash FROM jobs").fetchall()
        return [(r["uid"], r["description_hash"] or "") for r in rows]

    def stats(self) -> dict[str, Any]:
        with self.connect() as conn:
            total = conn.execute("SELECT COUNT(*) AS n FROM jobs").fetchone()["n"]
            companies = conn.execute("SELECT COUNT(DISTINCT company) AS n FROM jobs").fetchone()["n"]
            by_source = {
                r["source"]: r["n"]
                for r in conn.execute(
                    "SELECT source, COUNT(*) AS n FROM jobs GROUP BY source"
                ).fetchall()
            }
            runs = conn.execute("SELECT COUNT(*) AS n FROM runs").fetchone()["n"]
        return {
            "total_jobs": total,
            "distinct_companies": companies,
            "by_source": by_source,
            "total_runs": runs,
        }
