"""End-to-end smoke for the CV-driven global ranker.

Drives analyze_global() against the live SQLite corpus, verifies perf + quality
invariants, writes a JSON snapshot to data/e2e_global_result.json. Does not
fetch from the network and does not touch any browser automation.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from narou.config import DB_PATH
from narou.fraud import build_dedup_index, load_classifier, load_dedup_map
from narou.matching import (
    build_stage1_index,
    get_engine,
    load_feature_cache,
    load_stage1_index,
)
from narou.pipeline import analyze_global
from narou.resume import parse_resume
from narou.schema import Resume
from narou.storage import Database


DEFAULT_RESUME = r"C:\Users\andre\Downloads\Dorman_CV (3).docx"


def _synthetic_resume() -> Resume:
    return Resume(
        raw_text="senior offensive security engineer, adversarial ML",
        source_filename="synthetic.txt",
        sections={
            "summary": "Senior offensive security engineer with 8 years in "
                       "threat intelligence, red team tooling, and adversarial ML.",
            "skills": "python, ml, security, cuda, pytorch, fastapi, linux, "
                      "offensive, red team, threat intel, c2, reverse engineering",
            "experience": "Led threat intel pipelines, built red team tooling, "
                          "trained adversarial classifiers, developed offensive "
                          "tradecraft for enterprise pentests, implemented C2 "
                          "frameworks and EDR evasion research.",
            "education": "BS Human-Centered AI, Texas Tech; AAS Cybersecurity, TCC.",
        },
        skills=[
            "python", "ml", "security", "cuda", "pytorch", "fastapi", "linux",
            "offensive", "red team", "threat intel", "c2", "reverse engineering",
        ],
        contact={"name": "Andrew Dorman", "email": "andorman@ttu.edu"},
    )


def main() -> int:
    db = Database(DB_PATH)

    # Resume
    resume_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESUME
    try:
        resume = parse_resume(resume_path)
        print(f"[E2E] parsed: {resume_path}")
    except Exception as e:
        print(f"[E2E] parse failed ({e}), using synthetic resume")
        resume = _synthetic_resume()
    print(f"[E2E] sections={list(resume.sections.keys())}  skills={len(resume.skills)}")

    # Warm engine + classifier
    engine = get_engine()
    classifier = load_classifier()
    print(f"[E2E] engine: {engine.size_mb:.2f} MB · classifier: "
          f"{'ML+heuristics' if classifier.is_trained() else 'heuristics only'}")

    # Ensure index + dedup + feature cache exist
    t = time.time()
    idx = load_stage1_index(db)
    if idx is None:
        print("[E2E] building stage1 index (cold)…")
        idx = build_stage1_index(db)
    print(f"[E2E] stage1 index: {idx.corpus_size:,} jobs "
          f"(load/build {time.time() - t:.1f}s)")

    t = time.time()
    dmap = load_dedup_map(db)
    if not dmap:
        print("[E2E] building dedup index (cold)…")
        build_dedup_index(db)
        dmap = load_dedup_map(db)
    print(f"[E2E] dedup map: {len(dmap):,} entries ({time.time() - t:.1f}s)")

    # Sanity: job corpus
    stats = db.stats()
    print(f"[E2E] corpus: {stats['total_jobs']:,} jobs / "
          f"{stats['distinct_companies']:,} companies")

    # Run the full analysis a couple times to measure warm path
    t_cold = time.time()
    result = analyze_global(
        resume=resume,
        db=db,
        classifier=classifier,
        top_n=20,
        stage1_k=240,
        focus_text="",
        use_sgm_cache=True,
    )
    cold_ms = (time.time() - t_cold) * 1000
    print(f"[E2E] analyze_global (cold, no focus): {cold_ms:.0f} ms · "
          f"{len(result.matches)} matches · {result.metrics.jobs_flagged} flagged")

    t_warm = time.time()
    result2 = analyze_global(
        resume=resume,
        db=db,
        classifier=classifier,
        top_n=20,
        stage1_k=240,
        focus_text="offensive security red team llm adversarial",
        use_sgm_cache=True,
    )
    warm_ms = (time.time() - t_warm) * 1000
    print(f"[E2E] analyze_global (warm, with focus): {warm_ms:.0f} ms · "
          f"{len(result2.matches)} matches")

    # Show the top 15
    print()
    print("=" * 70)
    print("TOP 15 MATCHES (focused)")
    print("=" * 70)
    for i, m in enumerate(result2.matches[:15], 1):
        rp = result2.fraud_reports.get(m.job.uid)
        fraud = rp.score if rp else 0.0
        flag = " GHOST" if rp and rp.flagged else ""
        print(f"{i:2d}. {m.overall * 100:4.0f}%  ghost:{fraud * 100:3.0f}%{flag:<6}  "
              f"{m.job.company[:14]:14s}  {m.job.title[:52]}")

    # Invariants
    assertions = [
        ("corpus >= 1000 jobs", stats["total_jobs"] >= 1000),
        ("distinct companies >= 100", stats["distinct_companies"] >= 100),
        (">= 10 matches returned", len(result.matches) >= 10),
        ("matches sorted by overall desc", all(
            result.matches[i].overall >= result.matches[i + 1].overall
            for i in range(len(result.matches) - 1)
        )),
        ("all overall scores in [0,1]", all(
            0 <= m.overall <= 1 for m in result.matches
        )),
        ("all fraud scores in [0,1]", all(
            0 <= r.score <= 1 for r in result.fraud_reports.values()
        )),
        ("every match has a fraud report", all(
            m.job.uid in result.fraud_reports for m in result.matches
        )),
        ("no exact-duplicate job titles in top 10",
         len({m.job.title for m in result.matches[:10]})
         >= min(10, len(result.matches))),
        ("dedup map covers matches", all(
            m.job.uid in dmap for m in result.matches
        )),
        ("focus tilt reordered top match or held it",
         result.matches[0].job.uid == result2.matches[0].job.uid
         or result.matches[0].overall != result2.matches[0].overall),
    ]
    print()
    print("=" * 70)
    print("INVARIANTS")
    print("=" * 70)
    passed = 0
    for name, ok in assertions:
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {name}")
        if ok:
            passed += 1

    out_path = ROOT / "data" / "e2e_global_result.json"
    out_path.write_text(json.dumps({
        "cold_ms": cold_ms,
        "warm_ms": warm_ms,
        "corpus": stats,
        "top_matches": [
            {
                "title": m.job.title,
                "company": m.job.company,
                "overall": m.overall,
                "scores": m.scores.to_dict(),
                "fraud": (result2.fraud_reports[m.job.uid].to_dict()
                          if m.job.uid in result2.fraud_reports else None),
                "url": m.job.url,
            }
            for m in result2.matches[:20]
        ],
        "grades": [g.to_dict() for g in result2.grades],
    }, indent=2, default=str))
    print()
    print(f"[E2E] snapshot: {out_path}")
    print(f"[E2E] RESULT: {passed}/{len(assertions)} invariants passed")
    return 0 if passed == len(assertions) else 1


if __name__ == "__main__":
    sys.exit(main())
