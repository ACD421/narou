"""Parallel Greenhouse/Lever corpus crawler.

Reads a slug list from narou/data/boards_*.txt, fetches each board in parallel,
upserts jobs into SQLite, and tracks per-board health. Safe to run as a
background thread; does not block the UI.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..config import DATA_DIR, DB_PATH, JOB_CACHE_TTL
from ..schema import Job
from ..storage import Database
from .base import IngestionResult
from .greenhouse import GreenhouseAdapter
from .lever import LeverAdapter


SEED_GREENHOUSE = DATA_DIR / "boards_greenhouse_seed.txt"
SEED_LEVER = DATA_DIR / "boards_lever_seed.txt"

_crawl_lock = threading.Lock()
_crawl_state: dict = {
    "running": False,
    "started_at": 0.0,
    "finished_at": 0.0,
    "boards_total": 0,
    "boards_done": 0,
    "boards_ok": 0,
    "boards_failed": 0,
    "jobs_new": 0,
    "jobs_seen": 0,
    "last_summary": None,
    "last_error": "",
}


def crawl_state() -> dict:
    """Snapshot of the live crawl state (thread-safe read)."""
    with _crawl_lock:
        return dict(_crawl_state)


@dataclass
class CrawlResult:
    boards_total: int = 0
    boards_ok: int = 0
    boards_failed: int = 0
    jobs_seen: int = 0
    jobs_new_or_updated: int = 0
    elapsed_ms: float = 0.0
    failures: list[tuple[str, str, str]] = field(default_factory=list)  # (source, board, err)


def load_slugs(path: Path) -> list[str]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def _fetch_one_greenhouse(board: str, timeout: float) -> tuple[IngestionResult, float]:
    start = time.time()
    with GreenhouseAdapter(timeout=timeout) as adapter:
        result = adapter.fetch(board)
    return result, (time.time() - start) * 1000


def _fetch_one_lever(board: str, timeout: float) -> tuple[IngestionResult, float]:
    start = time.time()
    with LeverAdapter(timeout=timeout) as adapter:
        result = adapter.fetch(board)
    return result, (time.time() - start) * 1000


def crawl_corpus(
    db: Database | None = None,
    *,
    max_workers: int = 24,
    greenhouse_slugs: list[str] | None = None,
    lever_slugs: list[str] | None = None,
    progress_cb: Callable[[dict], None] | None = None,
    timeout: float = 12.0,
) -> CrawlResult:
    """Fetch all configured boards in parallel and upsert to the jobs table."""
    db = db or Database(DB_PATH)
    gh_slugs = greenhouse_slugs if greenhouse_slugs is not None else load_slugs(SEED_GREENHOUSE)
    lv_slugs = lever_slugs if lever_slugs is not None else load_slugs(SEED_LEVER)

    total = len(gh_slugs) + len(lv_slugs)
    res = CrawlResult(boards_total=total)
    t_start = time.time()

    with _crawl_lock:
        _crawl_state.update(
            running=True,
            started_at=t_start,
            finished_at=0.0,
            boards_total=total,
            boards_done=0,
            boards_ok=0,
            boards_failed=0,
            jobs_new=0,
            jobs_seen=0,
            last_error="",
        )

    def _record(src: str, board: str, result: IngestionResult, elapsed_ms: float) -> int:
        try:
            if result.ok:
                count = db.upsert_jobs(result.jobs)
                db.upsert_board_health(
                    source=src,
                    board=board,
                    ok=True,
                    status="ok",
                    jobs_seen=result.count,
                )
                return count
            else:
                db.upsert_board_health(
                    source=src,
                    board=board,
                    ok=False,
                    status="fail",
                    error=result.error,
                    jobs_seen=0,
                )
                return 0
        except Exception as e:
            db.upsert_board_health(
                source=src, board=board, ok=False, status="db_error", error=str(e)
            )
            return 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {}
        for slug in gh_slugs:
            futs[ex.submit(_fetch_one_greenhouse, slug, timeout)] = ("greenhouse", slug)
        for slug in lv_slugs:
            futs[ex.submit(_fetch_one_lever, slug, timeout)] = ("lever", slug)

        for fut in as_completed(futs):
            src, slug = futs[fut]
            try:
                result, elapsed_ms = fut.result()
            except Exception as e:
                res.boards_failed += 1
                res.failures.append((src, slug, f"exception: {e}"))
                try:
                    db.upsert_board_health(src, slug, ok=False, status="exception", error=str(e))
                except Exception:
                    pass
                with _crawl_lock:
                    _crawl_state["boards_done"] += 1
                    _crawl_state["boards_failed"] += 1
                if progress_cb:
                    try:
                        progress_cb(crawl_state())
                    except Exception:
                        pass
                continue

            added = _record(src, slug, result, elapsed_ms)
            if result.ok:
                res.boards_ok += 1
                res.jobs_seen += result.count
                res.jobs_new_or_updated += added
            else:
                res.boards_failed += 1
                res.failures.append((src, slug, result.error))

            with _crawl_lock:
                _crawl_state["boards_done"] += 1
                if result.ok:
                    _crawl_state["boards_ok"] += 1
                    _crawl_state["jobs_seen"] += result.count
                    _crawl_state["jobs_new"] += added
                else:
                    _crawl_state["boards_failed"] += 1
            if progress_cb:
                try:
                    progress_cb(crawl_state())
                except Exception:
                    pass

    res.elapsed_ms = (time.time() - t_start) * 1000
    db.crawl_meta_set("last_crawl_finished_at", str(time.time()))
    db.crawl_meta_set("last_crawl_boards_ok", str(res.boards_ok))
    db.crawl_meta_set("last_crawl_jobs_seen", str(res.jobs_seen))
    db.crawl_meta_set(
        "last_crawl_elapsed_ms", f"{res.elapsed_ms:.0f}"
    )

    with _crawl_lock:
        _crawl_state["running"] = False
        _crawl_state["finished_at"] = time.time()
        _crawl_state["last_summary"] = {
            "boards_ok": res.boards_ok,
            "boards_failed": res.boards_failed,
            "jobs_seen": res.jobs_seen,
            "jobs_new": res.jobs_new_or_updated,
            "elapsed_ms": res.elapsed_ms,
        }
    if progress_cb:
        try:
            progress_cb(crawl_state())
        except Exception:
            pass
    return res


def crawl_in_background(
    db: Database | None = None,
    *,
    max_workers: int = 12,
    min_interval_sec: float = JOB_CACHE_TTL,
    timeout: float = 12.0,
) -> threading.Thread | None:
    """Kick off a crawl on a daemon thread if none is running and the corpus is stale.

    Returns the thread if started, else None.
    """
    db = db or Database(DB_PATH)

    with _crawl_lock:
        if _crawl_state["running"]:
            return None

    last = db.crawl_meta_get("last_crawl_finished_at")
    if last:
        try:
            age = time.time() - float(last)
            if age < min_interval_sec:
                return None
        except ValueError:
            pass

    t = threading.Thread(
        target=crawl_corpus,
        kwargs={"db": db, "max_workers": max_workers, "timeout": timeout},
        daemon=True,
        name="narou-crawler",
    )
    t.start()
    return t
