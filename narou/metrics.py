from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class RunMetrics:
    boards_requested: int = 0
    boards_ok: int = 0
    boards_failed: int = 0
    jobs_ingested: int = 0
    jobs_parsed: int = 0
    jobs_matched: int = 0
    jobs_flagged: int = 0
    resume_parsed: bool = False
    resume_parse_error: str = ""
    time_to_first_result_ms: float = 0.0
    total_elapsed_ms: float = 0.0
    failures: list[dict] = field(default_factory=list)
    per_board: dict[str, dict] = field(default_factory=dict)

    _start_ts: float = field(default_factory=time.perf_counter)
    _first_result_ts: float | None = None

    def mark_first_result(self) -> None:
        if self._first_result_ts is None:
            self._first_result_ts = time.perf_counter()
            self.time_to_first_result_ms = (self._first_result_ts - self._start_ts) * 1000

    def finish(self) -> None:
        self.total_elapsed_ms = (time.perf_counter() - self._start_ts) * 1000

    @property
    def feed_uptime(self) -> float:
        if self.boards_requested == 0:
            return 0.0
        return self.boards_ok / self.boards_requested

    @property
    def parse_accuracy(self) -> float:
        if self.jobs_ingested == 0:
            return 0.0
        return self.jobs_parsed / self.jobs_ingested

    @property
    def fraud_flag_rate(self) -> float:
        if self.jobs_ingested == 0:
            return 0.0
        return self.jobs_flagged / self.jobs_ingested

    def to_dict(self) -> dict:
        return {
            "boards_requested": self.boards_requested,
            "boards_ok": self.boards_ok,
            "boards_failed": self.boards_failed,
            "feed_uptime": self.feed_uptime,
            "jobs_ingested": self.jobs_ingested,
            "jobs_parsed": self.jobs_parsed,
            "parse_accuracy": self.parse_accuracy,
            "jobs_matched": self.jobs_matched,
            "jobs_flagged": self.jobs_flagged,
            "fraud_flag_rate": self.fraud_flag_rate,
            "resume_parsed": self.resume_parsed,
            "resume_parse_error": self.resume_parse_error,
            "time_to_first_result_ms": self.time_to_first_result_ms,
            "total_elapsed_ms": self.total_elapsed_ms,
            "failures": self.failures,
            "per_board": self.per_board,
        }
