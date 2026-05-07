from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from ..config import STAGE1_MAX_CANDIDATES, STAGE1_MIN_CANDIDATES, STAGE1_TOPK_FRACTION
from ..schema import Job, Resume
from .aligner import (
    SectionScores,
    build_resume_cache,
    score_resume_vs_job,
)
from .index import Stage1Index, stage1_rank
from .sgm import (
    SGMEngine,
    TextFeatures,
    compute_features,
    fast_similarity_features,
    get_engine,
)


@dataclass
class RankedMatch:
    job: Job
    scores: SectionScores
    stage1_score: float

    @property
    def overall(self) -> float:
        return self.scores.overall


def _stage1_text(resume: Resume) -> str:
    parts = []
    if resume.sections.get("summary"):
        parts.append(resume.sections["summary"])
    if resume.sections.get("skills"):
        parts.append(resume.sections["skills"])
    elif resume.skills:
        parts.append(", ".join(resume.skills))
    if resume.sections.get("experience"):
        parts.append(resume.sections["experience"][:1500])
    if not parts:
        return resume.raw_text[:2500]
    return "\n".join(parts)[:2500]


def _job_stage1_text(job: Job) -> str:
    return f"{job.title}\n{(job.description or '')[:2500]}"


def _resume_stage1_query(resume: Resume) -> str:
    return _stage1_text(resume)


def _job_features(job: Job, cache: dict[str, TextFeatures] | None) -> TextFeatures:
    if cache is not None:
        f = cache.get(job.uid)
        if f is not None:
            return f
    f = compute_features(_job_stage1_text(job))
    if cache is not None:
        cache[job.uid] = f
    return f


def rank_jobs(
    resume: Resume,
    jobs: list[Job],
    engine: SGMEngine | None = None,
    top_n: int | None = None,
    feature_provider: Callable[[Job], TextFeatures] | None = None,
    feature_map: dict[str, TextFeatures] | None = None,
    focus_features: TextFeatures | None = None,
    focus_weight: float = 0.35,
) -> list[RankedMatch]:
    """Rank jobs against a resume.

    feature_provider: if supplied, returns precomputed TextFeatures per job (DB cache).
    focus_features / focus_weight: optional post-CV re-weight from a user 'focus' string.
    """
    if not jobs:
        return []
    engine = engine or get_engine()
    r_text = _stage1_text(resume)
    r_feats = compute_features(r_text)
    resume_cache = build_resume_cache(resume, engine)

    local_cache: dict[str, TextFeatures] = {}

    def _get_feats(job: Job) -> TextFeatures:
        if feature_map is not None:
            f = feature_map.get(job.uid)
            if f is not None and not f.is_empty:
                return f
        if feature_provider is not None:
            f = feature_provider(job)
            if f is not None and not f.is_empty:
                return f
        return _job_features(job, local_cache)

    stage1: list[tuple[Job, float, TextFeatures]] = []
    use_focus = focus_features is not None and not focus_features.is_empty
    for job in jobs:
        if not job.description:
            continue
        j_feats = _get_feats(job)
        base = fast_similarity_features(r_feats, j_feats)
        if use_focus:
            focus_sim = fast_similarity_features(focus_features, j_feats)
            score = (1.0 - focus_weight) * base + focus_weight * focus_sim
        else:
            score = base
        stage1.append((job, score, j_feats))

    stage1.sort(key=lambda p: -p[1])
    keep = max(STAGE1_MIN_CANDIDATES, int(len(stage1) * STAGE1_TOPK_FRACTION))
    keep = min(keep, STAGE1_MAX_CANDIDATES)
    survivors = stage1[:keep]

    ranked: list[RankedMatch] = []
    for job, s1_score, _feats in survivors:
        sec = score_resume_vs_job(
            resume, job, engine=engine, resume_cache=resume_cache, stage1_lexical=s1_score
        )
        if use_focus:
            sec.overall = (1.0 - focus_weight) * sec.overall + focus_weight * s1_score
        ranked.append(RankedMatch(job=job, scores=sec, stage1_score=s1_score))

    ranked.sort(key=lambda r: -r.overall)
    if top_n is not None:
        ranked = ranked[:top_n]
    return ranked


def rank_jobs_global(
    resume: Resume,
    index: Stage1Index,
    jobs_by_uid: dict[str, Job],
    engine: SGMEngine | None = None,
    focus_text: str = "",
    stage1_k: int = 120,
    top_n: int = 30,
    focus_weight: float = 0.35,
    sgm_loader=None,
    dedup_map: dict[str, dict] | None = None,
) -> list[RankedMatch]:
    """Fast global ranker: TF-IDF stage1 over pre-built index + SGM rerank.

    sgm_loader: optional callable(list[uid]) -> dict[uid, sgm_doc] that fetches
    precomputed SGM embeddings for stage1 survivors. If provided, rerank skips
    per-job tokenize+embed and drops to cosine-only work.
    """
    engine = engine or get_engine()
    resume_text = _stage1_text(resume)
    s1 = stage1_rank(
        index,
        resume_text=resume_text,
        focus_text=focus_text,
        top_k=stage1_k,
        focus_weight=focus_weight,
    )
    if not s1:
        return []

    resume_cache = build_resume_cache(resume, engine)

    # Dedup survivors by (uid, content_fp). We keep the first (highest s1) hit
    # per content fingerprint so near-identical reposts collapse.
    survivor_uids: list[str] = []
    seen_uids: set[str] = set()
    seen_content_fps: set[str] = set()
    s1_by_uid: dict[str, float] = {}
    for job_idx, s1_score in s1:
        uid = index.job_uids[job_idx]
        if uid in seen_uids:
            continue
        seen_uids.add(uid)
        if dedup_map is not None:
            dinfo = dedup_map.get(uid)
            if dinfo:
                cfp = dinfo.get("content_fp") or ""
                if cfp and cfp in seen_content_fps:
                    continue
                if cfp:
                    seen_content_fps.add(cfp)
        survivor_uids.append(uid)
        s1_by_uid[uid] = s1_score

    sgm_map: dict = {}
    if sgm_loader is not None:
        try:
            sgm_map = sgm_loader(survivor_uids) or {}
        except Exception:
            sgm_map = {}

    ranked: list[RankedMatch] = []
    for uid in survivor_uids:
        job = jobs_by_uid.get(uid)
        if job is None or not job.description:
            continue
        sec = score_resume_vs_job(
            resume,
            job,
            engine=engine,
            resume_cache=resume_cache,
            stage1_lexical=s1_by_uid[uid],
            job_sgm=sgm_map.get(uid),
        )
        ranked.append(RankedMatch(job=job, scores=sec, stage1_score=s1_by_uid[uid]))

    ranked.sort(key=lambda r: -r.overall)
    return ranked[:top_n]
