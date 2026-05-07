from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..schema import Job


@dataclass
class IngestionResult:
    source: str
    board: str
    jobs: list[Job] = field(default_factory=list)
    ok: bool = True
    error: str = ""
    elapsed_ms: float = 0.0
    from_cache: bool = False

    @property
    def count(self) -> int:
        return len(self.jobs)


class IngestionAdapter(Protocol):
    source: str

    def fetch(self, board: str) -> IngestionResult: ...
