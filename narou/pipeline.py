from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .config import (
    DEFAULT_GREENHOUSE_BOARDS,
    DEFAULT_LEVER_BOARDS,
    FRAUD_FLAG_THRESHOLD,
)
from .fraud import (
    FraudClassifier,
    FraudReport,
    build_dedup_index,
    load_classifier,
    load_dedup_map,
    score_jobs,
)
from .grading import CompanyGrade, grade_companies
from .ingestion import ADAPTERS, IngestionResult
from .matching import (
    RankedMatch,
    Stage1Index,
    build_stage1_index,
    get_engine,
    get_stage1_index,
    load_stage1_index,
    rank_jobs,
    rank_jobs_global,
)
from .matching.cache import load_sgm_blobs, populate_features
from .metrics import RunMetrics
from .schema import Job, Resume
from .storage import Database
from .suggestions import SuggestionReport, generate_suggestions


@dataclass
class AnalysisResult:
    resume: Resume
    matches: list[RankedMatch]
    fraud_reports: dict[str, FraudReport]
    suggestions: SuggestionReport
    grades: list[CompanyGrade]
    metrics: RunMetrics
    jobs_by_uid: dict[str, Job]

    def matches_with_fraud(self, fraud_threshold: float = FRAUD_FLAG_THRESHOLD) -> list[tuple[RankedMatch, FraudReport]]:
        out: list[tuple[RankedMatch, FraudReport]] = []
        for m in self.matches:
            rp = self.fraud_reports.get(m.job.uid)
            if rp is None:
                continue
            out.append((m, rp))
        return out


def _fetch_all(
    boards_by_source: dict[str, list[str]],
    metrics: RunMetrics,
) -> list[Job]:
    jobs: list[Job] = []
    for source, boards in boards_by_source.items():
        adapter_cls = ADAPTERS.get(source)
        if adapter_cls is None:
            continue
        with adapter_cls() as adapter:
            for board in boards:
                metrics.boards_requested += 1
                result: IngestionResult = adapter.fetch(board)
                metrics.per_board[f"{source}:{board}"] = {
                    "ok": result.ok,
                    "count": result.count,
                    "elapsed_ms": result.elapsed_ms,
                    "error": result.error,
                }
                if result.ok:
                    metrics.boards_ok += 1
                    metrics.jobs_ingested += result.count
                    metrics.jobs_parsed += result.count
                    jobs.extend(result.jobs)
                else:
                    metrics.boards_failed += 1
                    metrics.failures.append({
                        "source": source, "board": board, "error": result.error,
                    })
    return jobs


def analyze(
    resume: Resume,
    boards_by_source: dict[str, list[str]] | None = None,
    classifier: FraudClassifier | None = None,
    db: Database | None = None,
    top_n: int = 30,
) -> AnalysisResult:
    """Legacy board-scan pipeline (kept for tests and CLI use)."""
    metrics = RunMetrics()
    metrics.resume_parsed = True

    if not boards_by_source:
        boards_by_source = {
            "greenhouse": DEFAULT_GREENHOUSE_BOARDS,
            "lever": DEFAULT_LEVER_BOARDS,
        }

    jobs = _fetch_all(boards_by_source, metrics)

    if db is not None:
        try:
            db.upsert_jobs(jobs)
        except Exception as e:
            metrics.failures.append({"stage": "db_upsert", "error": str(e)})

    engine = get_engine()
    matches = rank_jobs(resume, jobs, engine=engine, top_n=top_n)
    metrics.jobs_matched = len(matches)
    if matches:
        metrics.mark_first_result()

    classifier = classifier or load_classifier()
    reports = score_jobs(jobs, classifier=classifier)
    fraud_by_uid = {r.job_uid: r for r in reports}
    metrics.jobs_flagged = sum(1 for r in reports if r.flagged)

    grades = grade_companies(jobs, reports)
    suggestions = generate_suggestions(resume, matches)

    metrics.finish()

    if db is not None:
        run_id = f"run_{int(datetime.now().timestamp())}"
        db.record_run(
            run_id=run_id,
            boards_scanned=metrics.boards_requested,
            jobs_ingested=metrics.jobs_ingested,
            jobs_matched=metrics.jobs_matched,
            jobs_flagged=metrics.jobs_flagged,
            elapsed_ms=metrics.total_elapsed_ms,
            notes=f"{len(boards_by_source)} source(s), {len(jobs)} jobs",
        )

    jobs_by_uid = {j.uid: j for j in jobs}

    return AnalysisResult(
        resume=resume,
        matches=matches,
        fraud_reports=fraud_by_uid,
        suggestions=suggestions,
        grades=grades,
        metrics=metrics,
        jobs_by_uid=jobs_by_uid,
    )


def analyze_global(
    resume: Resume,
    db: Database,
    *,
    classifier: FraudClassifier | None = None,
    top_n: int = 30,
    stage1_k: int = 200,
    focus_text: str = "",
    focus_weight: float = 0.35,
    use_sgm_cache: bool = True,
    jobs_by_uid: dict[str, Job] | None = None,
    stage1_index: Stage1Index | None = None,
    dedup_map: dict[str, dict] | None = None,
) -> AnalysisResult:
    """CV-driven global search over the full pre-crawled corpus.

    Does NOT fetch from the network. Ranks against the in-DB corpus using
    the pre-built TF-IDF index + SGM rerank + cross-company dedup.
    """
    metrics = RunMetrics()
    metrics.resume_parsed = True

    engine = get_engine()
    classifier = classifier or load_classifier()

    # Prefer caller-supplied caches (from Streamlit's @st.cache_resource);
    # otherwise fall back to on-demand load.
    index = stage1_index
    if index is None:
        index = load_stage1_index(db)
        if index is None:
            index = build_stage1_index(db)

    if jobs_by_uid is None:
        jobs = db.list_all_jobs()
        jobs_by_uid = {j.uid: j for j in jobs}
    metrics.jobs_ingested = len(jobs_by_uid)

    if dedup_map is None:
        try:
            dedup_map = load_dedup_map(db)
            if not dedup_map:
                build_dedup_index(db)
                dedup_map = load_dedup_map(db)
        except Exception as e:
            metrics.failures.append({"stage": "dedup", "error": str(e)})
            dedup_map = {}

    def _sgm_loader(uids: list[str]) -> dict:
        if not use_sgm_cache:
            return {}
        try:
            return load_sgm_blobs(db, uids)
        except Exception:
            return {}

    matches = rank_jobs_global(
        resume=resume,
        index=index,
        jobs_by_uid=jobs_by_uid,
        engine=engine,
        focus_text=focus_text,
        stage1_k=stage1_k,
        top_n=top_n,
        focus_weight=focus_weight,
        sgm_loader=_sgm_loader,
        dedup_map=dedup_map,
    )
    metrics.jobs_matched = len(matches)
    if matches:
        metrics.mark_first_result()

    # Fraud scoring runs over the matched set (not the entire 50k+ corpus)
    # since the user cares about the top results. Dedup info is plumbed in.
    match_jobs = [m.job for m in matches]
    reports = score_jobs(match_jobs, classifier=classifier, dedup_map=dedup_map)
    fraud_by_uid = {r.job_uid: r for r in reports}
    metrics.jobs_flagged = sum(1 for r in reports if r.flagged)

    grades = grade_companies(match_jobs, reports)
    suggestions = generate_suggestions(resume, matches)

    metrics.finish()

    if db is not None:
        run_id = f"run_{int(datetime.now().timestamp())}"
        try:
            db.record_run(
                run_id=run_id,
                boards_scanned=index.corpus_size if index else 0,
                jobs_ingested=metrics.jobs_ingested,
                jobs_matched=metrics.jobs_matched,
                jobs_flagged=metrics.jobs_flagged,
                elapsed_ms=metrics.total_elapsed_ms,
                notes=f"global, focus='{focus_text[:40]}', top_n={top_n}",
            )
        except Exception:
            pass

    return AnalysisResult(
        resume=resume,
        matches=matches,
        fraud_reports=fraud_by_uid,
        suggestions=suggestions,
        grades=grades,
        metrics=metrics,
        jobs_by_uid=jobs_by_uid,
    )
