from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from narou.config import FRAUD_MODEL_PATH
from narou.fraud.train import (
    find_labeled_csv,
    load_labeled_csv,
    train_models,
    weak_label_from_heuristics,
)
from narou.ingestion import GreenhouseAdapter


def collect_weak_labels(boards: list[str]) -> list:
    from narou.schema import Job
    all_jobs: list[Job] = []
    with GreenhouseAdapter() as adapter:
        for b in boards:
            r = adapter.fetch(b)
            if r.ok:
                all_jobs.extend(r.jobs)
                print(f"  {b}: {r.count} jobs")
            else:
                print(f"  {b}: FAIL ({r.error})")
    return all_jobs


def main():
    parser = argparse.ArgumentParser(description="Train the narou fraud classifier")
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to labeled CSV (columns: title, description, fraudulent)"
    )
    parser.add_argument(
        "--boards", nargs="+",
        default=["airbnb", "stripe", "databricks", "cloudflare", "gitlab"],
        help="Greenhouse boards for weak-label bootstrap",
    )
    parser.add_argument(
        "--output", type=str, default=str(FRAUD_MODEL_PATH),
        help="Output path for the trained classifier pickle",
    )
    args = parser.parse_args()

    data = None
    if args.csv:
        print(f"[1] Loading labeled CSV: {args.csv}")
        data = load_labeled_csv(args.csv)
        print(f"    loaded {len(data.features)} rows, {sum(data.labels)} positive")
    else:
        auto_csv = find_labeled_csv()
        if auto_csv is not None:
            print(f"[1] Auto-detected labeled CSV: {auto_csv}")
            data = load_labeled_csv(auto_csv)
            print(f"    loaded {len(data.features)} rows, {sum(data.labels)} positive")

    if data is None or len(data.features) < 20:
        print(f"[1] No labeled CSV found. Bootstrap from {len(args.boards)} live boards...")
        jobs = collect_weak_labels(args.boards)
        print(f"    ingested {len(jobs)} jobs")
        data = weak_label_from_heuristics(jobs)
        print(f"    weak-labeled {len(data.features)} jobs, {sum(data.labels)} positive")

    if len(data.features) < 20:
        print("ERROR: not enough training data. Try more boards or provide --csv.")
        sys.exit(1)

    print("[2] Training models...")
    classifier = train_models(data)

    print("[3] Metrics:")
    metrics = classifier.metrics or {}
    print(f"    source: {metrics.get('source')}")
    print(f"    train={metrics.get('n_train')} test={metrics.get('n_test')} pos_rate={metrics.get('pos_rate', 0):.2%}")
    for m in ("logistic", "random_forest"):
        if m in metrics:
            x = metrics[m]
            print(f"    {m:>13}: P={x['precision']:.2f} R={x['recall']:.2f} F1={x['f1']:.2f} AUC={x['roc_auc']:.2f}")

    print(f"[4] Saving to {args.output}")
    classifier.save(args.output)
    print("Done.")


if __name__ == "__main__":
    main()
