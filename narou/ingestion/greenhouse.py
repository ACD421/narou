from __future__ import annotations

import time

import httpx

from ..schema import Job
from ..utils import parse_iso_datetime, strip_html, timer
from .base import IngestionResult


GREENHOUSE_BASE = "https://boards-api.greenhouse.io/v1/boards"
USER_AGENT = "Narou/0.1 (+https://github.com/ACD421/narou)"


class GreenhouseAdapter:
    source = "greenhouse"

    def __init__(self, timeout: float = 15.0, client: httpx.Client | None = None):
        self._client = client or httpx.Client(
            timeout=timeout,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "GreenhouseAdapter":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def fetch(self, board: str) -> IngestionResult:
        url = f"{GREENHOUSE_BASE}/{board}/jobs?content=true"
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

        raw_jobs = data.get("jobs") or []
        parsed: list[Job] = []
        for rj in raw_jobs:
            job = self._parse(board, rj)
            if job is not None:
                parsed.append(job)
        result.jobs = parsed
        return result

    def _parse(self, board: str, rj: dict) -> Job | None:
        jid = rj.get("id")
        title = (rj.get("title") or "").strip()
        if jid is None or not title:
            return None
        location_obj = rj.get("location") or {}
        location = (location_obj.get("name") or "").strip() if isinstance(location_obj, dict) else ""
        departments = rj.get("departments") or []
        department = None
        if departments and isinstance(departments, list):
            first = departments[0]
            if isinstance(first, dict):
                department = first.get("name")
        description_html = rj.get("content") or ""
        description = strip_html(description_html)
        url = rj.get("absolute_url") or f"https://boards.greenhouse.io/{board}/jobs/{jid}"
        return Job(
            job_id=str(jid),
            source=self.source,
            company=board,
            title=title,
            location=location,
            department=department,
            description=description,
            url=url,
            posted_at=parse_iso_datetime(rj.get("first_published") or rj.get("updated_at")),
            updated_at=parse_iso_datetime(rj.get("updated_at")),
            raw=rj,
        )


def fetch_board(board: str) -> IngestionResult:
    with GreenhouseAdapter() as adapter:
        return adapter.fetch(board)
