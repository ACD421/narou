"""Sparse TF-IDF inverted index for stage-1 retrieval.

Builds once over the full corpus. At query time, a single resume→sparse vector
transform + sparse matmul returns top-K candidates in a few milliseconds,
independent of corpus size for realistic workloads.
"""
from __future__ import annotations

import pickle
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from ..config import CACHE_DIR, DB_PATH
from ..schema import Job
from ..storage import Database


_INDEX_FILE = CACHE_DIR / "stage1_index.npz"
_META_FILE = CACHE_DIR / "stage1_index.pkl"


_index_lock = threading.Lock()
_index_singleton: "Stage1Index | None" = None


@dataclass
class Stage1Index:
    job_uids: list[str]
    companies: list[str]
    vec_char: TfidfVectorizer
    vec_word: TfidfVectorizer
    mat_char: sparse.csr_matrix
    mat_word: sparse.csr_matrix
    built_at: float
    corpus_size: int

    @property
    def size(self) -> int:
        return self.corpus_size


def _job_stage1_text(job: Job) -> str:
    return f"{job.title} {job.title} {job.department or ''} {(job.description or '')[:2500]}"


def build_stage1_index(
    db: Database | None = None,
    persist: bool = True,
) -> Stage1Index | None:
    db = db or Database(DB_PATH)
    t0 = time.time()
    jobs = db.list_all_jobs()
    if not jobs:
        return None

    texts = [_job_stage1_text(j) for j in jobs]
    job_uids = [j.uid for j in jobs]
    companies = [j.company for j in jobs]

    vec_char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 4),
        min_df=3,
        max_df=0.85,
        max_features=20_000,
        sublinear_tf=True,
        norm="l2",
        lowercase=True,
        strip_accents="unicode",
    )
    mat_char = vec_char.fit_transform(texts)

    vec_word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.85,
        max_features=15_000,
        sublinear_tf=True,
        norm="l2",
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        token_pattern=r"\b[a-zA-Z][a-zA-Z0-9+\-_.]+\b",
    )
    mat_word = vec_word.fit_transform(texts)

    idx = Stage1Index(
        job_uids=job_uids,
        companies=companies,
        vec_char=vec_char,
        vec_word=vec_word,
        mat_char=mat_char.tocsr(),
        mat_word=mat_word.tocsr(),
        built_at=time.time(),
        corpus_size=len(jobs),
    )

    if persist:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(_INDEX_FILE.with_suffix(".char.npz"), idx.mat_char)
        sparse.save_npz(_INDEX_FILE.with_suffix(".word.npz"), idx.mat_word)
        with open(_META_FILE, "wb") as f:
            pickle.dump(
                {
                    "job_uids": job_uids,
                    "companies": companies,
                    "vec_char": vec_char,
                    "vec_word": vec_word,
                    "built_at": idx.built_at,
                    "corpus_size": idx.corpus_size,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        db.crawl_meta_set("stage1_index_built_at", str(idx.built_at))
        db.crawl_meta_set("stage1_index_size", str(idx.corpus_size))

    return idx


def load_stage1_index(db: Database | None = None) -> Stage1Index | None:
    char_path = _INDEX_FILE.with_suffix(".char.npz")
    word_path = _INDEX_FILE.with_suffix(".word.npz")
    if not char_path.exists() or not word_path.exists() or not _META_FILE.exists():
        return None
    mat_char = sparse.load_npz(char_path).tocsr()
    mat_word = sparse.load_npz(word_path).tocsr()
    with open(_META_FILE, "rb") as f:
        meta = pickle.load(f)
    return Stage1Index(
        job_uids=meta["job_uids"],
        companies=meta["companies"],
        vec_char=meta["vec_char"],
        vec_word=meta["vec_word"],
        mat_char=mat_char,
        mat_word=mat_word,
        built_at=meta["built_at"],
        corpus_size=meta["corpus_size"],
    )


def get_stage1_index(
    db: Database | None = None, force_rebuild: bool = False
) -> Stage1Index | None:
    global _index_singleton
    with _index_lock:
        if _index_singleton is not None and not force_rebuild:
            return _index_singleton
        db = db or Database(DB_PATH)
        if not force_rebuild:
            idx = load_stage1_index(db)
            if idx is not None:
                _index_singleton = idx
                return idx
        built = build_stage1_index(db)
        if built is not None:
            _index_singleton = built
        return _index_singleton


def invalidate_stage1_index() -> None:
    global _index_singleton
    with _index_lock:
        _index_singleton = None


def stage1_rank(
    index: Stage1Index,
    resume_text: str,
    focus_text: str = "",
    top_k: int = 120,
    focus_weight: float = 0.35,
    char_weight: float = 0.55,
) -> list[tuple[int, float]]:
    """Return the top-K (job_idx, score) candidates from a sparse TF-IDF query."""
    if not resume_text:
        return []
    # Resume char and word vectors (shape (1, V_char) and (1, V_word))
    rc = index.vec_char.transform([resume_text])
    rw = index.vec_word.transform([resume_text])
    # Sparse cosine similarity = dot product since matrices are l2-normalized
    scores = char_weight * (index.mat_char @ rc.T).toarray().ravel()
    scores += (1.0 - char_weight) * (index.mat_word @ rw.T).toarray().ravel()

    if focus_text:
        fc = index.vec_char.transform([focus_text])
        fw_ = index.vec_word.transform([focus_text])
        focus_scores = char_weight * (index.mat_char @ fc.T).toarray().ravel()
        focus_scores += (1.0 - char_weight) * (index.mat_word @ fw_.T).toarray().ravel()
        scores = (1.0 - focus_weight) * scores + focus_weight * focus_scores

    if top_k >= len(scores):
        order = np.argsort(-scores)
    else:
        part = np.argpartition(-scores, top_k)[:top_k]
        order = part[np.argsort(-scores[part])]
    return [(int(i), float(scores[i])) for i in order]


def index_stats(index: Stage1Index | None = None) -> dict:
    if index is None:
        return {"built": False}
    return {
        "built": True,
        "corpus_size": index.corpus_size,
        "built_at": index.built_at,
        "age_sec": time.time() - index.built_at,
        "char_vocab": len(index.vec_char.vocabulary_),
        "word_vocab": len(index.vec_word.vocabulary_),
        "nnz_char": int(index.mat_char.nnz),
        "nnz_word": int(index.mat_word.nnz),
    }
