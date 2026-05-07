from __future__ import annotations

from datetime import datetime, timezone

import httpx

from ..schema import Job
from ..utils import strip_html, timer
from .base import IngestionResult


LEVER_BASE = "https://api.lever.co/v0/postings"
USER_AGENT = "Narou/0.1 (+https://github.com/ACD421/narou)"


class LeverAdapter:
    source = "lever"

    def __init__(self, timeout: float = 15.0, client: httpx.Client | None = None):
        self._client = client or httpx.Client(
            timeout=timeout,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "LeverAdapter":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def fetch(self, board: str) -> IngestionResult:
        url = f"{LEVER_BASE}/{board}?mode=json"
        result = IngestionResult(source=self.source, board=board)
        with timer() as t:
            try:
                resp = self._client.get(url)
            except httpx.HTTPError as e:
                result.ok = False
                result.error = f"network: {e}"
                result.elapsed_ms = t["elapsed"] * 1000
                return result
        result.elapsed_ms = t["elapsed"] * 1000

        if resp.status_code == 404:
            result.ok = False
            result.error = "board not found"
            return result
        if resp.status_code != 200:
            result.ok = False
            result.error = f"http {resp.status_code}"
            return result

        try:
            data = resp.json()
        except ValueError as e:
            result.ok = False
            result.error = f"json parse: {e}"
            return result

        raw_jobs = data if isinstance(data, list) else []
        parsed: list[Job] = []
        for rj in raw_jobs:
            job = self._parse(board, rj)
            if job is not None:
                parsed.append(job)
        result.jobs = parsed
        return result

    def _parse(self, board: str, rj: dict) -> Job | None:
        jid = rj.get("id")
        title = (rj.get("text") or "").strip()
        if not jid or not title:
            return None

        categories = rj.get("categories") or {}
        location = (categories.get("location") or "").strip() if isinstance(categories, dict) else ""
        department = categories.get("team") if isinstance(categories, dict) else None

        parts: list[str] = []
        desc_plain = rj.get("descriptionPlain")
        desc_html = rj.get("description")
        if desc_plain:
            parts.append(desc_plain)
        elif desc_html:
            parts.append(strip_html(desc_html))
        for lst in rj.get("lists") or []:
            if not isinstance(lst, dict):
                continue
            head = lst.get("text") or ""
            content = lst.get("content") or ""
            if head:
                parts.append(head)
            if content:
                parts.append(strip_html(content))
        closing = rj.get("additionalPlain") or rj.get("additional")
        if closing:
            parts.append(strip_html(closing) if "<" in closing else closing)
        description = "\n\n".join(p for p in parts if p).strip()

        posted_ms = rj.get("createdAt")
        posted_at = None
        if isinstance(posted_ms, (int, float)) and posted_ms > 0:
            posted_at = datetime.fromtimestamp(posted_ms / 1000, tz=timezone.utc)

        return Job(
            job_id=str(jid),
            source=self.source,
            company=board,
            title=title,
            location=location,
            department=department if isinstance(department, str) else None,
            description=description,
            url=rj.get("hostedUrl") or rj.get("applyUrl") or "",
            posted_at=posted_at,
            updated_at=None,
            raw=rj,
        )


def fetch_board(board: str) -> IngestionResult:
    with LeverAdapter() as adapter:
        return adapter.fetch(board)
