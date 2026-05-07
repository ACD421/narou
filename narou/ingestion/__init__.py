from .base import IngestionAdapter, IngestionResult
from .crawler import (
    CrawlResult,
    crawl_corpus,
    crawl_in_background,
    crawl_state,
    load_slugs,
    SEED_GREENHOUSE,
    SEED_LEVER,
)
from .greenhouse import GreenhouseAdapter
from .lever import LeverAdapter

ADAPTERS: dict[str, type[IngestionAdapter]] = {
    "greenhouse": GreenhouseAdapter,
    "lever": LeverAdapter,
}

__all__ = [
    "IngestionAdapter",
    "IngestionResult",
    "GreenhouseAdapter",
    "LeverAdapter",
    "ADAPTERS",
    "CrawlResult",
    "crawl_corpus",
    "crawl_in_background",
    "crawl_state",
    "load_slugs",
    "SEED_GREENHOUSE",
    "SEED_LEVER",
]
