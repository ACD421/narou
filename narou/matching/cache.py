"""Job feature cache -- persistent + in-process.

Populate on corpus change; load once per process. Bulk load is fast enough
that queries see sub-millisecond per-job feature lookups.
"""
from __future__ import annotations

import pickle
import threading
import time
from typing import Iterable

from ..schema import Job
from ..storage import Database
from .sgm import SGMEngine, TextFeatures, compute_features, get_engine


_FEATURES_TABLE_VERSION = 3


_cache_lock = threading.Lock()
_feature_cache: dict[str, TextFeatures] | None = None
_cache_loaded_at: float = 0.0
_cache_job_count: int = 0


def _job_stage1_text(job: Job) -> str:
    return f"{job.title}\n{(job.description or '')[:2500]}"


def compute_and_pack(job: Job) -> tuple[str, str, bytes]:
    feats = compute_features(_job_stage1_text(job))
    blob = pickle.dumps(
        (list(feats.stems), list(feats.char3), list(feats.words)),
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    return job.uid, job.description_hash or "", blob


def compute_sgm_pack(job: Job, engine: SGMEngine) -> bytes:
    """Pre-embed a job into two SGM docs (body + header) as a single pickle blob."""
    job_text = f"{job.title}\n{job.department or ''}\n{job.description or ''}"[:2500]
    job_header = f"{job.title} {job.department or ''}".strip()
    body_doc = engine.precompute_doc(job_text) if job_text else None
    header_doc = engine.precompute_doc(job_header) if job_header else None
    return pickle.dumps(
        {"body": body_doc, "header": header_doc},
        protocol=pickle.HIGHEST_PROTOCOL,
    )


def unpack_sgm(blob: bytes) -> dict:
    try:
        data = pickle.loads(blob)
    except Exception:
        return {}
    # Lazily populate _features on old blobs pickled before the lex fast path
    # was added. One tokenization per side per job is still cheap, and avoids
    # re-populating 53k SGM blobs.
    for key in ("body", "header"):
        doc = data.get(key)
        if isinstance(doc, dict) and "_features" not in doc:
            doc["_features"] = compute_features(doc.get("_text") or "")
    return data


def _ensure_schema(db: Database) -> None:
    """Ensure job_features has features_blob + sgm_blob columns."""
    with db.connect() as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(job_features)").fetchall()}
        if "features_blob" not in cols:
            conn.execute("ALTER TABLE job_features ADD COLUMN features_blob BLOB")
        if "sgm_blob" not in cols:
            conn.execute("ALTER TABLE job_features ADD COLUMN sgm_blob BLOB")


def populate_features(
    db: Database,
    batch_size: int = 500,
    only_missing: bool = True,
    include_sgm: bool = True,
) -> dict:
    """Compute lexical features (and optionally SGM embeds) for every job
    and persist to job_features.

    If only_missing, skip jobs whose description_hash matches the stored one.
    Returns stats.
    """
    _ensure_schema(db)
    engine = get_engine() if include_sgm else None
    t0 = time.time()
    total = 0
    updated = 0
    skipped = 0

    with db.connect() as conn:
        stored = {
            row["uid"]: (row["description_hash"] or "", row["sgm_blob"] is not None)
            for row in conn.execute(
                "SELECT uid, description_hash, sgm_blob FROM job_features"
            ).fetchall()
        }

    batch: list[tuple[str, str, bytes, bytes | None]] = []

    def _flush(batch_: list[tuple[str, str, bytes, bytes | None]]) -> None:
        if not batch_:
            return
        now = time.time()
        with db.connect() as conn:
            rows = [(uid, dhash, now, fblob, sblob) for uid, dhash, fblob, sblob in batch_]
            conn.executemany(
                """
                INSERT INTO job_features(uid, description_hash, computed_at, features_blob, sgm_blob)
                VALUES(?,?,?,?,?)
                ON CONFLICT(uid) DO UPDATE SET
                    description_hash=excluded.description_hash,
                    computed_at=excluded.computed_at,
                    features_blob=excluded.features_blob,
                    sgm_blob=COALESCE(excluded.sgm_blob, job_features.sgm_blob)
                """,
                rows,
            )

    offset = 0
    while True:
        with db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY uid LIMIT ? OFFSET ?",
                (batch_size, offset),
            ).fetchall()
        if not rows:
            break
        offset += len(rows)
        for r in rows:
            total += 1
            uid = r["uid"]
            dhash = r["description_hash"] or ""
            prev = stored.get(uid)
            same_hash = prev and prev[0] == dhash and dhash
            has_sgm = prev and prev[1]
            need_features = not (only_missing and same_hash)
            need_sgm = include_sgm and not (only_missing and same_hash and has_sgm)
            if not need_features and not need_sgm:
                skipped += 1
                continue
            job = Database._row_to_job(r)
            try:
                uid_, dhash_, fblob = compute_and_pack(job)
                sblob = compute_sgm_pack(job, engine) if need_sgm and engine else None
                batch.append((uid_, dhash_, fblob, sblob))
                updated += 1
            except Exception:
                continue
            if len(batch) >= batch_size:
                _flush(batch)
                batch = []
    _flush(batch)

    db.crawl_meta_set("features_populated_at", str(time.time()))
    db.crawl_meta_set("features_version", str(_FEATURES_TABLE_VERSION))
    elapsed = time.time() - t0
    return {
        "total_jobs": total,
        "updated": updated,
        "skipped": skipped,
        "elapsed_sec": elapsed,
    }


def load_sgm_blobs(db: Database, uids: list[str]) -> dict[str, dict]:
    """Load SGM precomputed docs for a set of job uids (e.g., stage1 survivors)."""
    if not uids:
        return {}
    out: dict[str, dict] = {}
    with db.connect() as conn:
        CHUNK = 500
        for i in range(0, len(uids), CHUNK):
            chunk = uids[i:i + CHUNK]
            placeholders = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"SELECT uid, sgm_blob FROM job_features WHERE uid IN ({placeholders}) AND sgm_blob IS NOT NULL",
                chunk,
            ).fetchall()
            for r in rows:
                try:
                    out[r["uid"]] = unpack_sgm(r["sgm_blob"])
                except Exception:
                    continue
    return out


def _unpack_blob(blob: bytes) -> TextFeatures:
    stems, c3, words = pickle.loads(blob)
    return TextFeatures(
        stems=frozenset(stems),
        char3=frozenset(c3),
        words=frozenset(words),
    )


def load_feature_cache(db: Database, force: bool = False) -> dict[str, TextFeatures]:
    """Load all job features into an in-process dict keyed by uid.

    The dict is cached at module level; subsequent calls are O(1) unless
    the corpus has grown (then we incrementally top it up).
    """
    global _feature_cache, _cache_loaded_at, _cache_job_count

    with _cache_lock:
        if _feature_cache is None or force:
            t = time.time()
            cache: dict[str, TextFeatures] = {}
            with db.connect() as conn:
                rows = conn.execute(
                    "SELECT uid, features_blob FROM job_features WHERE features_blob IS NOT NULL"
                ).fetchall()
            for r in rows:
                try:
                    cache[r["uid"]] = _unpack_blob(r["features_blob"])
                except Exception:
                    continue
            _feature_cache = cache
            _cache_loaded_at = time.time()
            _cache_job_count = len(cache)
        elif _feature_cache is not None:
            # Top up with any new jobs whose features landed since last load.
            with db.connect() as conn:
                total = conn.execute(
                    "SELECT COUNT(*) AS n FROM job_features WHERE features_blob IS NOT NULL"
                ).fetchone()["n"]
            if total > _cache_job_count:
                known = set(_feature_cache.keys())
                with db.connect() as conn:
                    rows = conn.execute(
                        "SELECT uid, features_blob FROM job_features WHERE features_blob IS NOT NULL"
                    ).fetchall()
                for r in rows:
                    if r["uid"] not in known:
                        try:
                            _feature_cache[r["uid"]] = _unpack_blob(r["features_blob"])
                        except Exception:
                            continue
                _cache_job_count = len(_feature_cache)
        return _feature_cache


def cache_stats() -> dict:
    with _cache_lock:
        return {
            "entries": _cache_job_count,
            "loaded_at": _cache_loaded_at,
            "age_sec": (time.time() - _cache_loaded_at) if _cache_loaded_at else 0,
        }


def invalidate_cache() -> None:
    global _feature_cache, _cache_job_count, _cache_loaded_at
    with _cache_lock:
        _feature_cache = None
        _cache_job_count = 0
        _cache_loaded_at = 0.0
