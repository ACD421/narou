from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from narou.config import DB_PATH
from narou.fraud import load_classifier
from narou.matching import get_engine
from narou.pipeline import analyze
from narou.resume import parse_resume
from narou.storage import Database


def main():
    if len(sys.argv) < 2:
        resume_path = r"C:\Users\andre\Downloads\Dorman_CV (3).docx"
    else:
        resume_path = sys.argv[1]

    boards = {
        "greenhouse": ["cloudflare", "databricks", "stripe"],
        "lever": ["palantir"],
    }

    print(f"[E2E] resume: {resume_path}")
    resume = parse_resume(resume_path)
    print(f"[E2E] parsed sections: {list(resume.sections.keys())}")
    print(f"[E2E] skills: {len(resume.skills)}")
    print(f"[E2E] contact: {resume.contact.get('name','?')} | {resume.contact.get('email','?')}")

    print("[E2E] warming SGM engine...")
    engine = get_engine()
    classifier = load_classifier()
    print(f"[E2E] engine: {engine.size_mb:.2f} MB, {len(engine.feat_names)} features")
    print(f"[E2E] classifier: {'ML+heuristics' if classifier.is_trained() else 'heuristics only'}")

    print(f"[E2E] boards: {boards}")
    db = Database(DB_PATH)

    result = analyze(
        resume=resume,
        boards_by_source=boards,
        classifier=classifier,
        db=db,
        top_n=25,
    )

    m = result.metrics
    print()
    print("=" * 60)
    print("METRICS")
    print("=" * 60)
    print(f"  Feed uptime:       {m.feed_uptime*100:.0f}% ({m.boards_ok}/{m.boards_requested} boards)")
    print(f"  Jobs ingested:     {m.jobs_ingested}")
    print(f"  Jobs flagged:      {m.jobs_flagged} ({m.fraud_flag_rate*100:.1f}%)")
    print(f"  Jobs ranked:       {m.jobs_matched}")
    print(f"  First result in:   {m.time_to_first_result_ms:.0f} ms")
    print(f"  Total elapsed:     {m.total_elapsed_ms:.0f} ms ({m.total_elapsed_ms/1000:.1f}s)")

    print()
    print("=" * 60)
    print("TOP 10 MATCHES")
    print("=" * 60)
    for i, (match, fraud) in enumerate(result.matches_with_fraud()[:10], 1):
        flag = " GHOST" if fraud.flagged else ""
        print(f"  {i:2d}. {match.overall*100:5.1f}%  ghost={fraud.score*100:4.0f}%  "
              f"[{match.job.company:>12}] {match.job.title[:50]}{flag}")

    print()
    print("=" * 60)
    print("COMPANY GRADES")
    print("=" * 60)
    for g in result.grades:
        print(f"  [{g.letter}] {g.company:<15} vol={g.posting_volume:<4} "
              f"flag={g.flagged_rate*100:>4.0f}%  median_age={g.median_days_active:.0f}d")

    print()
    print("=" * 60)
    print("RESUME SUGGESTIONS")
    print("=" * 60)
    sug = result.suggestions
    print(f"  {sug.headline}")
    print(f"  {sug.summary}")
    print()
    print("  Top missing keywords:")
    for k, c in sug.top_missing_keywords[:6]:
        print(f"    - {k} ({c} matches)")
    print("  Section advice:")
    for section, advice in sug.section_advice.items():
        print(f"    [{section}] {advice[:160]}")

    print()
    print("=" * 60)
    print("INVARIANTS")
    print("=" * 60)
    assertions = [
        ("matches have overall scores in [0,1]", all(0 <= m.overall <= 1 for m in result.matches)),
        ("matches sorted by overall descending", all(
            result.matches[i].overall >= result.matches[i+1].overall
            for i in range(len(result.matches)-1)
        )),
        ("every match has fraud report", all(
            m.job.uid in result.fraud_reports for m in result.matches
        )),
        ("all fraud scores in [0,1]", all(
            0 <= r.score <= 1 for r in result.fraud_reports.values()
        )),
        ("grades cover all companies with matches", len({m.job.company for m in result.matches}) <= len({g.company for g in result.grades})),
        ("metrics feed_uptime in [0,1]", 0 <= m.feed_uptime <= 1),
        ("time_to_first_result > 0 when matches found", m.time_to_first_result_ms > 0 if result.matches else True),
    ]
    passed = 0
    for name, ok in assertions:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if ok:
            passed += 1

    out_path = ROOT / "data" / "e2e_last_result.json"
    out_path.write_text(json.dumps({
        "metrics": result.metrics.to_dict(),
        "top_matches": [
            {
                "title": m.job.title,
                "company": m.job.company,
                "overall": m.overall,
                "scores": m.scores.to_dict(),
                "fraud": result.fraud_reports[m.job.uid].to_dict() if m.job.uid in result.fraud_reports else None,
            }
            for m in result.matches[:20]
        ],
        "grades": [g.to_dict() for g in result.grades],
        "suggestions": result.suggestions.to_dict(),
    }, indent=2, default=str))
    print()
    print(f"[E2E] Result saved to {out_path}")

    print()
    print("=" * 60)
    print(f"RESULT: {passed}/{len(assertions)} invariants passed")
    print("=" * 60)
    return 0 if passed == len(assertions) else 1


if __name__ == "__main__":
    sys.exit(main())
