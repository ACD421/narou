"""Cross-company near-dup detection for job postings.

Two layers:
1. Content fingerprint (first ~120 normalized words) catches identical reposts
   and cross-board duplicates.
2. Title fingerprint (normalized title + company) catches same-role re-listings.

Populated as a one-shot batch over the corpus; stored in the job_dedup table.
Used by both the ranker (to collapse duplicates in the top-K output) and by
the fraud pipeline (cluster_size becomes a repost-count feature).
"""
from __future__ import annotations

import hashlib
import re
import time
from collections import defaultdict

from ..storage import Database


_WORD = re.compile(r"\b[a-z]+\b")

_DEDUP_SCHEMA = """
CREATE TABLE IF NOT EXISTS job_dedup (
    uid TEXT PRIMARY KEY,
    content_fp TEXT NOT NULL,
    title_fp TEXT NOT NULL,
    cluster_size INTEGER NOT NULL DEFAULT 1,
    title_cluster_size INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_dedup_content ON job_dedup(content_fp);
CREATE INDEX IF NOT EXISTS idx_dedup_title ON job_dedup(title_fp);
"""


def normalize_words(text: str) -> list[str]:
    return [w for w in _WORD.findall((text or "").lower()) if len(w) > 2]


def content_fingerprint(description: str, window: int = 120) -> str:
    words = normalize_words(description)[:window]
    if not words:
        return ""
    return hashlib.md5(" ".join(words).encode("utf-8")).hexdigest()


def title_fingerprint(title: str, department: str | None = None) -> str:
    t = " ".join(normalize_words(title or "")) + "|" + " ".join(normalize_words(department or ""))
    return hashlib.md5(t.encode("utf-8")).hexdigest()


def _ensure_schema(db: Database) -> None:
    with db.connect() as conn:
        conn.executescript(_DEDUP_SCHEMA)


def build_dedup_index(db: Database, batch_size: int = 2000) -> dict:
    """Compute content + title fingerprints for every job and populate job_dedup."""
    _ensure_schema(db)
    t0 = time.time()

    entries: list[tuple[str, str, str]] = []  # (uid, content_fp, title_fp)
    content_counts: dict[str, int] = defaultdict(int)
    title_counts: dict[str, int] = defaultdict(int)

    offset = 0
    while True:
        with db.connect() as conn:
            rows = conn.execute(
                "SELECT uid, title, department, description FROM jobs ORDER BY uid LIMIT ? OFFSET ?",
                (batch_size, offset),
            ).fetchall()
        if not rows:
            break
        offset += len(rows)
        for r in rows:
            uid = r["uid"]
            cfp = content_fingerprint(r["description"] or "")
            tfp = title_fingerprint(r["title"] or "", r["department"])
            entries.append((uid, cfp, tfp))
            if cfp:
                content_counts[cfp] += 1
            title_counts[tfp] += 1

    with db.connect() as conn:
        conn.execute("DELETE FROM job_dedup")
        conn.executemany(
            """
            INSERT INTO job_dedup(uid, content_fp, title_fp, cluster_size, title_cluster_size)
            VALUES(?,?,?,?,?)
            """,
            [
                (
                    uid,
                    cfp,
                    tfp,
                    content_counts.get(cfp, 1) if cfp else 1,
                    title_counts.get(tfp, 1),
                )
                for uid, cfp, tfp in entries
            ],
        )

    distinct_content = len(content_counts)
    distinct_title = len(title_counts)
    max_content_cluster = max(content_counts.values(), default=1)
    max_title_cluster = max(title_counts.values(), default=1)
    # Histograms for sanity
    content_dups = sum(1 for v in content_counts.values() if v > 1)
    title_dups = sum(1 for v in title_counts.values() if v > 1)

    db.crawl_meta_set("dedup_built_at", str(time.time()))
    db.crawl_meta_set("dedup_corpus_size", str(len(entries)))

    return {
        "corpus_size": len(entries),
        "distinct_content_fps": distinct_content,
        "distinct_title_fps": distinct_title,
        "content_dup_clusters": content_dups,
        "title_dup_clusters": title_dups,
        "max_content_cluster": max_content_cluster,
        "max_title_cluster": max_title_cluster,
        "elapsed_sec": time.time() - t0,
    }


def load_dedup_map(db: Database) -> dict[str, dict]:
    """Return {uid: {content_fp, title_fp, cluster_size, title_cluster_size}}."""
    out: dict[str, dict] = {}
    with db.connect() as conn:
        for r in conn.execute(
            "SELECT uid, content_fp, title_fp, cluster_size, title_cluster_size FROM job_dedup"
        ).fetchall():
            out[r["uid"]] = {
                "content_fp": r["content_fp"],
                "title_fp": r["title_fp"],
                "cluster_size": r["cluster_size"],
                "title_cluster_size": r["title_cluster_size"],
            }
    return out
