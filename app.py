from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Trying to unpickle estimator", module="sklearn")

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from narou.config import DB_PATH, FRAUD_FLAG_THRESHOLD
from narou.fraud import build_dedup_index, load_classifier, load_dedup_map
from narou.ingestion import crawl_in_background, crawl_state
from narou.matching import (
    build_stage1_index,
    get_engine,
    index_stats,
    load_stage1_index,
)
from narou.pipeline import analyze_global
from narou.resume import ParseError, parse_bytes
from narou.storage import Database


st.set_page_config(
    page_title="Narou -- AI Job Search Assistant",
    page_icon=":mag:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- Color helpers ----------

_GREEN = "#22c55e"
_YELLOW = "#eab308"
_ORANGE = "#f97316"
_RED = "#ef4444"


def _match_color(pct: float) -> str:
    """Higher match = better. Green >= 50, yellow 35-49, red < 35."""
    if pct >= 50:
        return _GREEN
    if pct >= 35:
        return _YELLOW
    return _RED


def _risk_color(pct: float) -> str:
    """Lower risk = better. Green < 20, yellow 20-59, orange >= 60."""
    if pct < 20:
        return _GREEN
    if pct < 60:
        return _YELLOW
    return _ORANGE


def _bar_css(value: float, max_val: float, color: str) -> str:
    """CSS for a colored bar fill inside a table cell."""
    w = min(100.0, (value / max(max_val, 1)) * 100)
    return (
        f"background: linear-gradient(90deg, {color}40 {w:.0f}%, transparent {w:.0f}%);"
        f" color: {color}; font-weight: 600;"
    )


def _style_match_col(s: pd.Series) -> list[str]:
    return [_bar_css(v, 100, _match_color(v)) for v in s]


def _style_risk_col(s: pd.Series) -> list[str]:
    return [_bar_css(v, 100, _risk_color(v)) for v in s]


def _ghost_threshold_label(val: float) -> tuple[str, str]:
    """Return (description, hex_color) for the current ghost-risk threshold."""
    if val <= 0.25:
        return "Very strict -- flags anything remotely suspicious", _RED
    if val <= 0.40:
        return "Strict -- catches most fakes, may hide a few real ones", _ORANGE
    if val <= 0.60:
        return "Balanced -- good default for most searches", _GREEN
    if val <= 0.75:
        return "Lenient -- only flags obvious fakes", _YELLOW
    return "Very lenient -- almost nothing gets flagged", _RED


def _search_depth_label(val: int) -> tuple[str, str]:
    """Return (description, hex_color) for the rerank depth."""
    if val <= 100:
        return "Fast -- quick results, might miss niche matches", _YELLOW
    if val <= 300:
        return "Balanced -- good speed and coverage", _GREEN
    if val <= 750:
        return "Thorough -- slower, catches more long-tail matches", _YELLOW
    if val <= 1500:
        return "Deep -- reranks a large slice of the corpus", _ORANGE
    return "Maximum -- reranks nearly everything, slowest", _RED


# ---------- Cached singletons ----------


@st.cache_resource(show_spinner=False)
def get_db() -> Database:
    return Database(DB_PATH)


@st.cache_resource(show_spinner=False)
def load_engine_cached():
    return get_engine()


@st.cache_resource(show_spinner=False)
def load_classifier_cached():
    return load_classifier()


def corpus_is_empty() -> bool:
    try:
        return get_db().stats().get("total_jobs", 0) == 0
    except Exception:
        return True


@st.cache_resource(show_spinner="Loading match index…")
def load_index_cached():
    db = get_db()
    try:
        idx = load_stage1_index(db)
    except Exception:
        idx = None
    if idx is None:
        if corpus_is_empty():
            return None
        idx = build_stage1_index(db)
    return idx


@st.cache_resource(show_spinner=False)
def load_dedup_map_cached():
    db = get_db()
    try:
        m = load_dedup_map(db)
    except Exception:
        m = {}
    if not m:
        if corpus_is_empty():
            return {}
        build_dedup_index(db)
        m = load_dedup_map(db)
    return m


@st.cache_resource(show_spinner="Loading job corpus…")
def load_jobs_cached():
    db = get_db()
    jobs = db.list_all_jobs()
    return {j.uid: j for j in jobs}


def _invalidate_corpus_caches() -> None:
    """Drop cached index/jobs/dedup so next query rebuilds from fresh DB state."""
    load_index_cached.clear()
    load_dedup_map_cached.clear()
    load_jobs_cached.clear()


# ---------- State helpers ----------


def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("resume", None)
    ss.setdefault("result", None)
    ss.setdefault("analyzing", False)
    ss.setdefault("focus", "")
    ss.setdefault("top_n", 20)
    ss.setdefault("fraud_threshold", FRAUD_FLAG_THRESHOLD)
    ss.setdefault("stage1_k", 150)

    # Auto-crawl on first launch when the corpus is empty.
    if not ss.get("_auto_crawl_done") and corpus_is_empty():
        ss["_auto_crawl_done"] = True
        db = get_db()
        t = crawl_in_background(db, min_interval_sec=0)
        if t is not None:
            ss["_auto_crawl_active"] = True


# ---------- Sidebar (upload + focus + run) ----------


def _render_sidebar() -> bool:
    with st.sidebar:
        st.title("Narou")
        st.caption("Smart matching. Real opportunities. No more fake jobs.")
        st.caption("HCAI 4350 • Team Narou")

        st.markdown("---")

        st.subheader("1. Your resume")
        uploaded = st.file_uploader(
            "PDF or DOCX",
            type=["pdf", "docx"],
            accept_multiple_files=False,
            key="resume_upload",
            help="Your resume never leaves this machine. Parsing, matching and "
                 "fraud scoring all run locally.",
        )
        if uploaded is not None:
            try:
                resume = parse_bytes(uploaded.getvalue(), uploaded.name)
                st.session_state.resume = resume
                name = resume.contact.get("name") or uploaded.name
                st.success(f"Parsed: **{name}**")
                cols = st.columns(2)
                cols[0].metric("Sections", len(resume.sections))
                cols[1].metric("Skills", len(resume.skills))
                with st.expander("Resume preview"):
                    if resume.contact.get("name"):
                        st.write(f"**{resume.contact['name']}**")
                    if resume.contact.get("email"):
                        st.caption(resume.contact["email"])
                    for sec, text in resume.sections.items():
                        if sec.startswith("_"):
                            continue
                        st.markdown(f"**{sec.title()}**")
                        st.text(text[:500] + ("…" if len(text) > 500 else ""))
            except ParseError as e:
                st.error(f"Parse failed: {e}")
                st.session_state.resume = None

        st.markdown("---")

        st.subheader("2. Focus (optional)")
        st.text_input(
            "Tilt the ranking",
            key="focus",
            placeholder="e.g. remote senior LLM red team, healthcare UX, Rust systems",
            help="Free-text nudge. Your CV drives matches; this re-weights the top "
                 "ranking toward whatever matters most right now. Leave blank to "
                 "let the CV speak for itself.",
        )

        st.markdown("---")

        st.subheader("3. Settings")
        st.slider(
            "Results to show",
            5, 50,
            key="top_n",
            help="How many of your best-matched jobs to display after analysis.",
        )
        st.slider(
            "Fake-job sensitivity",
            0.10, 0.90,
            key="fraud_threshold",
            step=0.05,
            help="How aggressively to flag suspicious postings. Lower = stricter "
                 "(flags more), higher = more lenient (shows more results).",
        )
        thresh_desc, thresh_color = _ghost_threshold_label(
            st.session_state.fraud_threshold
        )
        st.markdown(
            f'<span style="color:{thresh_color}; font-size:0.85em;">'
            f"{thresh_desc}</span>",
            unsafe_allow_html=True,
        )
        with st.expander("Advanced"):
            st.slider(
                "Rerank depth",
                50, 2000,
                key="stage1_k",
                step=50,
                help="Every job in the corpus is searched instantly. This controls "
                     "how many top candidates get a deep comparison to your resume. "
                     "Higher = more thorough but slower.",
            )
            depth_desc, depth_color = _search_depth_label(
                st.session_state.stage1_k
            )
            st.markdown(
                f'<span style="color:{depth_color}; font-size:0.85em;">'
                f"{depth_desc}</span>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        st.subheader("4. Analyze")
        can_run = st.session_state.resume is not None and not st.session_state.analyzing
        label = "Analyze" if can_run else (
            "Upload a resume to begin" if st.session_state.resume is None else "Analyzing…"
        )
        run = st.button(
            label,
            type="primary",
            use_container_width=True,
            disabled=not can_run,
        )

        st.markdown("---")
        _render_corpus_status()
        return run


def _render_corpus_status() -> None:
    db = get_db()
    try:
        stats = db.stats()
    except Exception:
        stats = {"total_jobs": 0, "distinct_companies": 0}

    st.subheader("Corpus")
    cols = st.columns(2)
    cols[0].metric("Jobs", f"{stats.get('total_jobs', 0):,}")
    cols[1].metric("Companies", f"{stats.get('distinct_companies', 0):,}")

    try:
        idx = load_index_cached()
        idx_info = index_stats(idx)
        age = int(idx_info.get("age_sec", 0))
        st.caption(
            f"Index built · {idx_info.get('corpus_size', 0):,} jobs · "
            f"{age // 60} min old"
        )
    except Exception as e:
        st.caption(f"Index: not yet built ({e})")

    # Background crawler status
    cs = crawl_state()
    if cs["running"]:
        pct = cs["boards_done"] / max(1, cs["boards_total"])
        st.progress(
            min(1.0, pct),
            text=f"Crawling {cs['boards_done']}/{cs['boards_total']} boards…",
        )
    elif cs["last_summary"]:
        s = cs["last_summary"]
        st.caption(
            f"Last crawl: {s['boards_ok']} ok · {s['jobs_new']:,} new · "
            f"{s['elapsed_ms'] / 1000:.1f}s"
        )

    is_empty = corpus_is_empty()
    btn_label = "Seed corpus now" if is_empty else "Refresh corpus now"
    if st.button(btn_label, use_container_width=True, type="primary" if is_empty else "secondary"):
        t = crawl_in_background(db, min_interval_sec=0)
        if t is None:
            st.warning("A crawl is already running.")
        else:
            st.info(
                "Crawl started. First run takes ~1-2 minutes and "
                "populates ~10-20k jobs from ~300 live Greenhouse boards."
            )
            _invalidate_corpus_caches()


# ---------- Main body ----------


def _render_header() -> None:
    st.title("Narou -- AI Job Search Assistant")
    st.caption(
        "Your resume, the whole Greenhouse corpus, one click. "
        "Runs entirely on this machine -- no API keys, nothing leaves the box."
    )
    if corpus_is_empty():
        st.info(
            "**First run.** The job corpus is empty. Click **Seed corpus now** in "
            "the sidebar to crawl ~1,200 Greenhouse boards in parallel -- this "
            "takes about a minute on a warm network and runs in the background "
            "while you upload your resume."
        )


def _run_analysis() -> None:
    resume = st.session_state.resume
    if resume is None:
        st.warning("Upload a resume first.")
        return

    if corpus_is_empty():
        st.warning(
            "The job corpus is empty. Click **Seed corpus now** in the sidebar "
            "to crawl Greenhouse for the first time."
        )
        return

    engine = load_engine_cached()
    classifier = load_classifier_cached()
    db = get_db()

    # Warm caches up front so the spinner is on *our* work.
    stage1_index = load_index_cached()
    dedup_map = load_dedup_map_cached()
    jobs_by_uid = load_jobs_cached()

    if stage1_index is None or not jobs_by_uid:
        st.warning(
            "Corpus is still warming up. Wait for the crawler to finish, then "
            "click Analyze again."
        )
        return

    st.session_state.analyzing = True
    t0 = time.time()
    try:
        with st.spinner("Ranking CV against the global corpus…"):
            result = analyze_global(
                resume=resume,
                db=db,
                classifier=classifier,
                top_n=st.session_state.top_n,
                stage1_k=st.session_state.stage1_k,
                focus_text=st.session_state.focus.strip(),
                use_sgm_cache=True,
                jobs_by_uid=jobs_by_uid,
                stage1_index=stage1_index,
                dedup_map=dedup_map,
            )
        st.session_state.result = result
        st.session_state.last_rank_ms = (time.time() - t0) * 1000
    finally:
        st.session_state.analyzing = False


def _render_matches(result) -> None:
    if not result or not result.matches:
        st.info("Upload a resume and click Analyze to see your best-matched jobs.")
        return

    threshold = st.session_state.fraud_threshold
    rows = []
    for i, m in enumerate(result.matches):
        rp = result.fraud_reports.get(m.job.uid)
        fraud = rp.score if rp else 0.0
        rows.append({
            "Rank": i + 1,
            "Match %": round(m.overall * 100, 1),
            "Ghost %": round(fraud * 100, 1),
            "Risk": "Ghost" if fraud >= threshold else "OK",
            "Title": m.job.title,
            "Company": m.job.company,
            "Location": m.job.location or "-",
            "Source": m.job.source,
            "Days open": m.job.days_active if m.job.days_active is not None else "-",
            "URL": m.job.url or "",
        })
    df = pd.DataFrame(rows)

    display_cols = [
        "Rank", "Match %", "Ghost %", "Risk", "Title", "Company",
        "Location", "Source", "Days open", "URL",
    ]
    styled = (
        df[display_cols]
        .style
        .apply(_style_match_col, subset=["Match %"])
        .apply(_style_risk_col, subset=["Ghost %"])
        .format({"Match %": "{:.0f}%", "Ghost %": "{:.0f}%"})
    )
    st.dataframe(
        styled, hide_index=True, use_container_width=True, height=540,
        column_config={"URL": st.column_config.LinkColumn("Open")},
    )

    st.subheader("Inspect a match")
    titles = [f"#{r['Rank']} · {r['Title']} ({r['Company']})" for r in rows]
    choice = st.selectbox(
        "Pick a job", list(range(len(rows))), format_func=lambda i: titles[i]
    )
    if choice is None:
        return
    match = result.matches[choice]
    job = match.job
    rp = result.fraud_reports.get(job.uid)

    left, right = st.columns([2, 1])
    with left:
        st.markdown(f"### {job.title}")
        st.markdown(f"**{job.company}** · {job.location or 'Location not specified'}")
        if job.url:
            st.markdown(f"[Open posting]({job.url})")
        st.markdown("---")
        st.text_area(
            "Description",
            (job.description or "")[:8000],
            height=340,
            label_visibility="collapsed",
        )
    with right:
        overall_pct = match.overall * 100
        ghost_pct = (rp.score if rp else 0) * 100
        st.markdown(
            f'<div style="font-size:1.1em; font-weight:700; '
            f'color:{_match_color(overall_pct)};">Overall match: '
            f'{overall_pct:.0f}%</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="font-size:1.1em; font-weight:700; '
            f'color:{_risk_color(ghost_pct)};">Ghost risk: '
            f'{ghost_pct:.0f}%</div>',
            unsafe_allow_html=True,
        )
        st.markdown("**Section breakdown**")
        s = match.scores
        for label, val in [
            ("Title", s.title), ("Skills", s.skills),
            ("Summary", s.summary), ("Experience", s.experience),
            ("Lexical", s.lexical),
        ]:
            pct = val * 100
            c = _match_color(pct)
            st.markdown(
                f'<div style="margin:2px 0;">{label}: '
                f'<span style="color:{c}; font-weight:600;">'
                f'{pct:.0f}%</span></div>',
                unsafe_allow_html=True,
            )
        if s.matched_keywords:
            st.markdown("**Your skills found in this posting**")
            st.write(", ".join(s.matched_keywords[:12]))
        if s.missing_keywords:
            with st.expander("Your skills not mentioned in this posting"):
                st.write(", ".join(s.missing_keywords[:15]))
        if rp and rp.reasons:
            st.markdown("**Why it looks suspicious**")
            for r in rp.reasons:
                st.write(f"- {r}")


def _render_insights(result) -> None:
    if not result or not result.suggestions:
        st.info("Run an analysis to see resume insights.")
        return
    sug = result.suggestions
    st.markdown(f"### {sug.headline}")
    st.info(sug.summary)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Missing keywords** -- high-signal terms in your top matches, "
                    "but not on your resume")
        if sug.top_missing_keywords:
            for kw, count in sug.top_missing_keywords:
                st.write(f"- `{kw}` ({count} jobs)")
        else:
            st.caption("No significant gaps found.")
    with c2:
        st.markdown("**Strong keywords** -- already aligned")
        if sug.strong_keywords:
            for kw, count in sug.strong_keywords:
                st.write(f"- `{kw}` ({count} jobs)")
        else:
            st.caption("No strong matches surfaced yet.")

    st.markdown("---")
    st.markdown("### Section advice")
    if sug.section_advice:
        for section, advice in sug.section_advice.items():
            st.markdown(f"**{section.title()}**")
            st.write(advice)
    else:
        st.caption("Your resume sections match the target roles well.")

    if sug.rephrase_hints:
        st.markdown("### Concrete rephrase hints")
        for h in sug.rephrase_hints:
            st.write(f"- {h}")

    if sug.common_themes:
        st.markdown("### Themes in your target roles")
        st.write(" · ".join(sug.common_themes))


def _render_grades(result) -> None:
    if not result or not result.grades:
        st.info("Run an analysis to see company trust grades.")
        return
    rows = []
    for g in result.grades:
        rows.append({
            "Grade": g.letter,
            "Company": g.company,
            "Volume": g.posting_volume,
            "Flagged %": round(g.flagged_rate * 100, 1),
            "Median age (days)": round(g.median_days_active, 0),
            "Repost %": round(g.repost_rate * 100, 1),
            "Reasons": " · ".join(g.reasons) if g.reasons else "",
        })
    df = pd.DataFrame(rows)
    styled = (
        df.style
        .apply(_style_risk_col, subset=["Flagged %"])
        .apply(_style_risk_col, subset=["Repost %"])
        .format({"Flagged %": "{:.0f}%", "Repost %": "{:.0f}%"})
    )
    st.dataframe(styled, hide_index=True, use_container_width=True)


def _render_metrics(result) -> None:
    st.markdown("### This run")
    cols = st.columns(4)
    if result:
        m = result.metrics
        last_ms = st.session_state.get("last_rank_ms", m.total_elapsed_ms)
        cols[0].metric("Rank time", f"{last_ms:.0f} ms")
        cols[1].metric("Candidates", m.jobs_ingested)
        cols[2].metric("Flagged", m.jobs_flagged,
                       delta=f"{m.fraud_flag_rate * 100:.0f}%")
        cols[3].metric("Top N", len(result.matches))
    else:
        for c in cols:
            c.metric("-", "-")

    st.markdown("### Corpus health")
    db = get_db()
    try:
        with db.connect() as conn:
            total_boards = conn.execute("SELECT COUNT(*) FROM board_health").fetchone()[0]
            ok_boards = conn.execute(
                "SELECT COUNT(*) FROM board_health WHERE consecutive_failures = 0"
            ).fetchone()[0]
            total_jobs = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            feat_jobs = conn.execute(
                "SELECT COUNT(*) FROM job_features WHERE features_blob IS NOT NULL"
            ).fetchone()[0]
            sgm_jobs = conn.execute(
                "SELECT COUNT(*) FROM job_features WHERE sgm_blob IS NOT NULL"
            ).fetchone()[0]
    except Exception as e:
        st.error(f"Could not read corpus metrics: {e}")
        return

    cols = st.columns(4)
    cols[0].metric("Boards attempted", total_boards)
    cols[1].metric(
        "Feed uptime",
        f"{(ok_boards / max(1, total_boards)) * 100:.0f}%",
        delta=f"{ok_boards} ok",
    )
    cols[2].metric(
        "Features cached",
        f"{(feat_jobs / max(1, total_jobs)) * 100:.0f}%",
        delta=f"{feat_jobs:,} jobs",
    )
    cols[3].metric(
        "SGM cached",
        f"{(sgm_jobs / max(1, total_jobs)) * 100:.0f}%",
        delta=f"{sgm_jobs:,} jobs",
    )

    st.markdown("### Run history")
    try:
        runs = db.recent_runs(limit=10)
    except Exception as e:
        runs = []
        st.caption(f"Could not load run history: {e}")
    if runs:
        rdf = pd.DataFrame(runs)
        rdf["started_at"] = pd.to_datetime(rdf["started_at"], unit="s")
        display = [
            "started_at", "boards_scanned", "jobs_ingested", "jobs_matched",
            "jobs_flagged", "elapsed_ms", "notes",
        ]
        st.dataframe(rdf[display], hide_index=True, use_container_width=True)

    st.markdown("### Engine")
    engine = load_engine_cached()
    classifier = load_classifier_cached()
    cols = st.columns(3)
    cols[0].metric("SGM size", f"{engine.size_mb:.2f} MB")
    cols[1].metric("SGM features", len(engine.feat_names))
    cols[2].metric(
        "Fraud classifier",
        "trained" if classifier.is_trained() else "heuristics only",
    )

    st.markdown("### Retrieval index")
    try:
        idx = load_index_cached()
        info = index_stats(idx)
        cols = st.columns(4)
        cols[0].metric("Corpus", f"{info['corpus_size']:,}")
        cols[1].metric("Char vocab", f"{info['char_vocab']:,}")
        cols[2].metric("Word vocab", f"{info['word_vocab']:,}")
        cols[3].metric("Non-zeros", f"{(info['nnz_char'] + info['nnz_word']) / 1e6:.1f}M")
    except Exception as e:
        st.caption(f"Index unavailable: {e}")

    st.markdown("### Dedup")
    try:
        dmap = load_dedup_map_cached()
        content_clusters = {}
        for v in dmap.values():
            fp = v.get("content_fp") or ""
            if fp:
                content_clusters[fp] = content_clusters.get(fp, 0) + 1
        if content_clusters:
            dup_count = sum(1 for c in content_clusters.values() if c > 1)
            max_cluster = max(content_clusters.values())
            cols = st.columns(3)
            cols[0].metric("Distinct postings", f"{len(content_clusters):,}")
            cols[1].metric("Duplicate clusters", f"{dup_count:,}")
            cols[2].metric("Largest repost cluster", f"{max_cluster}")
    except Exception as e:
        st.caption(f"Dedup unavailable: {e}")


# ---------- Main ----------


_DISCLAIMER = (
    "Narou is designed to assist job seekers in navigating job opportunities, "
    "not to make decisions on their behalf. Results are generated using automated "
    "matching and should be treated as guidance only. Fraud and ghost-job scores "
    "are statistical estimates, not guarantees. Final decisions should always "
    "depend on your own judgment and independent verification of each opportunity."
)


def main() -> None:
    _init_state()
    run = _render_sidebar()
    _render_header()

    if run:
        _run_analysis()

    tab_matches, tab_insights, tab_grades, tab_metrics = st.tabs(
        ["Matches", "Resume Insights", "Company Grades", "System Metrics"]
    )
    result = st.session_state.result
    with tab_matches:
        _render_matches(result)
    with tab_insights:
        _render_insights(result)
    with tab_grades:
        _render_grades(result)
    with tab_metrics:
        _render_metrics(result)

    st.markdown("---")
    st.caption(_DISCLAIMER)


if __name__ == "__main__":
    main()
