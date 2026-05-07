from __future__ import annotations

import hashlib
import html
import re
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator


_HTML_TAG = re.compile(r"<[^>]+>")
_MULTI_WS = re.compile(r"\s+")
_SENT_SPLIT = re.compile(r"(?<=[\.\!\?])\s+(?=[A-Z])")


def strip_html(text: str) -> str:
    if not text:
        return ""
    cleaned = html.unescape(text)
    cleaned = _HTML_TAG.sub(" ", cleaned)
    cleaned = html.unescape(cleaned)
    return _MULTI_WS.sub(" ", cleaned).strip()


def sentence_split(text: str, min_len: int = 10) -> list[str]:
    if not text:
        return []
    normalized = _MULTI_WS.sub(" ", text.replace("\n", " ")).strip()
    sents = _SENT_SPLIT.split(normalized)
    return [s.strip() for s in sents if len(s.strip()) >= min_len]


def bullet_split(text: str, min_len: int = 5) -> list[str]:
    if not text:
        return []
    lines = re.split(r"[\n\u2022\u2023\u25E6\u2043\u00b7]|(?:^|\n)\s*[\*\-]\s+", text)
    return [ln.strip() for ln in lines if len(ln.strip()) >= min_len]


def short_hash(text: str, n: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:n]


def parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        v = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(v)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    state = {"elapsed": 0.0}
    start = time.perf_counter()
    try:
        yield state
    finally:
        state["elapsed"] = time.perf_counter() - start
