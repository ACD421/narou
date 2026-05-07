"""End-to-end headless smoke test for Narou.

1. Parse Andrew's CV (Dorman_CV_v4.docx)
2. Crawl 5 Greenhouse boards: scopely, cloudflare, stripe, figma, discord
3. Build stage1 index + dedup over that corpus
4. Run analyze_global against it
5. Report: job counts, match counts, top 3, errors/warnings
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message="Trying to unpickle estimator", module="sklearn")

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from narou.resume.parser import parse_resume
from narou.ingestion.greenhouse import GreenhouseAdapter
from narou.fraud import build_dedup_index, load_classifier, load_dedup_map
from narou.matching import build_stage1_index, get_engine
from narou.pipeline import analyze_global
from narou.storage import Database

# ── Config ──────────────────────────────────────────────────────────────
CV_PATH = Path(r"C:\Users\andre\Downloads\Dorman_CV_v4.docx")
BOARDS = ["scopely", "cloudflare", "stripe", "figma", "discord"]
SMOKE_DB = ROOT / "data" / "smoke_final.sqlite"


def main() -> int:
    errors: list[str] = []

    # ── Step 1: Parse resume ────────────────────────────────────────────
    print("=" * 70)
    print("STEP 1: Parse resume")
    print("=" * 70)
    try:
        resume = parse_resume(CV_PATH)
        print(f"  File:     {CV_PATH.name}")
        print(f"  Sections: {list(resume.sections.keys())}")
        print(f"  Skills:   {len(resume.skills)} extracted")
        print(f"  Contact:  {resume.contact}")
    except Exception as e:
        msg = f"Resume parse failed: {e}"
        print(f"  ERROR: {msg}")
        errors.append(msg)
        return 1

    # ── Step 2: Crawl 5 greenhouse boards ───────────────────────────────
    print()
    print("=" * 70)
    print("STEP 2: Crawl greenhouse boards")
    print("=" * 70)

    all_jobs = []
    crawl_warnings = []
    with GreenhouseAdapter(timeout=30.0) as adapter:
        for board in BOARDS:
            t0 = time.time()
            result = adapter.fetch(board)
            elapsed = (time.time() - t0) * 1000
            if result.ok:
                print(f"  {board:14s}  {result.count:4d} jobs  ({elapsed:.0f} ms)")
                all_jobs.extend(result.jobs)
            else:
                warn = f"{board}: {result.error}"
                crawl_warnings.append(warn)
                print(f"  {board:14s}  FAIL  {result.error}  ({elapsed:.0f} ms)")

    total_crawled = len(all_jobs)
    print(f"\n  Total crawled: {total_crawled} jobs")
    if crawl_warnings:
        print(f"  Warnings: {len(crawl_warnings)}")
        for w in crawl_warnings:
            errors.append(f"Crawl warning: {w}")

    if total_crawled == 0:
        msg = "No jobs crawled -- cannot continue"
        print(f"  FATAL: {msg}")
        errors.append(msg)
        return 1

    # ── Step 3: Build corpus in temp DB ─────────────────────────────────
    print()
    print("=" * 70)
    print("STEP 3: Build corpus + index")
    print("=" * 70)

    # Use a separate DB so we don't touch the production corpus
    if SMOKE_DB.exists():
        SMOKE_DB.unlink()
    db = Database(SMOKE_DB)
    inserted = db.upsert_jobs(all_jobs)
    print(f"  Upserted {inserted} jobs into {SMOKE_DB.name}")

    # Build stage1 index (persist=False to avoid overwriting prod cache)
    t0 = time.time()
    idx = build_stage1_index(db, persist=False)
    idx_ms = (time.time() - t0) * 1000
    if idx is None:
        msg = "build_stage1_index returned None"
        print(f"  ERROR: {msg}")
        errors.append(msg)
        return 1
    print(f"  Stage1 index: {idx.corpus_size} jobs, built in {idx_ms:.0f} ms")

    # Build dedup index
    t0 = time.time()
    build_dedup_index(db)
    dedup_map = load_dedup_map(db)
    dedup_ms = (time.time() - t0) * 1000
    print(f"  Dedup map: {len(dedup_map)} entries ({dedup_ms:.0f} ms)")

    # ── Step 4: Run analyze_global ──────────────────────────────────────
    print()
    print("=" * 70)
    print("STEP 4: analyze_global")
    print("=" * 70)

    classifier = load_classifier()
    engine = get_engine()
    print(f"  Engine: {engine.size_mb:.2f} MB")
    print(f"  Classifier: {'ML+heuristics' if classifier.is_trained() else 'heuristics only'}")

    t0 = time.time()
    result = analyze_global(
        resume=resume,
        db=db,
        classifier=classifier,
        top_n=30,
        stage1_k=200,
        focus_text="",
        use_sgm_cache=False,
        stage1_index=idx,
        dedup_map=dedup_map,
    )
    analysis_ms = (time.time() - t0) * 1000

    n_matches = len(result.matches)
    n_flagged = result.metrics.jobs_flagged
    print(f"  Matches returned: {n_matches}")
    print(f"  Fraud-flagged:    {n_flagged}")
    print(f"  Elapsed:          {analysis_ms:.0f} ms")

    if result.metrics.failures:
        for f in result.metrics.failures:
            w = f"Pipeline failure: {f}"
            print(f"  WARNING: {w}")
            errors.append(w)

    # ── Step 5: Report ──────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"  Jobs crawled:    {total_crawled}")
    print(f"  Jobs in corpus:  {idx.corpus_size}")
    print(f"  Matches:         {n_matches}")
    print(f"  Flagged (ghost): {n_flagged}")
    print()

    if n_matches > 0:
        print("  TOP 3 MATCHES:")
        print(f"  {'#':>3s}  {'Score':>6s}  {'Company':14s}  Title")
        print("  " + "-" * 64)
        for i, m in enumerate(result.matches[:3], 1):
            print(f"  {i:3d}  {m.overall*100:5.1f}%  {m.job.company[:14]:14s}  {m.job.title[:40]}")
        print()

        # Validation checks
        sorted_ok = all(
            result.matches[i].overall >= result.matches[i + 1].overall
            for i in range(len(result.matches) - 1)
        )
        scores_ok = all(0 <= m.overall <= 1 for m in result.matches)
        fraud_ok = all(
            0 <= r.score <= 1 for r in result.fraud_reports.values()
        )
        has_fraud = all(
            m.job.uid in result.fraud_reports for m in result.matches
        )

        checks = [
            ("Matches sorted desc", sorted_ok),
            ("All scores in [0,1]", scores_ok),
            ("All fraud scores in [0,1]", fraud_ok),
            ("Every match has fraud report", has_fraud),
            ("At least 1 match", n_matches >= 1),
        ]
        print("  CHECKS:")
        for name, ok in checks:
            tag = "PASS" if ok else "FAIL"
            print(f"    [{tag}] {name}")
            if not ok:
                errors.append(f"Check failed: {name}")
    else:
        errors.append("No matches returned")

    # ── Summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    if errors:
        print(f"DONE with {len(errors)} error(s)/warning(s):")
        for e in errors:
            print(f"  - {e}")
    else:
        print("DONE -- all clear")
    print("=" * 70)

    # Cleanup temp DB
    try:
        db._conn.close() if db._conn else None
        SMOKE_DB.unlink(missing_ok=True)
    except Exception:
        pass

    return 1 if any("FAIL" in e or "FATAL" in e or "ERROR" in e for e in errors) else 0


if __name__ == "__main__":
    sys.exit(main())
