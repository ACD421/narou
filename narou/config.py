from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
SGM_DIR = MODELS_DIR / "sgm"
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
LABELED_DIR = DATA_DIR / "labeled"
SAMPLES_DIR = DATA_DIR / "samples"
DB_PATH = DATA_DIR / "jobs.sqlite"

FRAUD_MODEL_PATH = MODELS_DIR / "fraud_classifier.joblib"

# Cache TTL for job fetches (seconds). Live feeds refresh within this window.
JOB_CACHE_TTL = 6 * 3600

# Default board suggestions shown in the UI. Users can override.
DEFAULT_GREENHOUSE_BOARDS = [
    "airbnb", "stripe", "figma", "notion", "gitlab",
    "databricks", "discord", "plaid", "vercel", "cloudflare",
]
DEFAULT_LEVER_BOARDS = [
    "palantir", "leverdemo",
]

# Fraud threshold: above this, flag as suspicious.
FRAUD_FLAG_THRESHOLD = 0.55

# Two-stage retrieval: stage 1 keeps top K% of jobs for SGM rerank.
STAGE1_TOPK_FRACTION = 0.15
STAGE1_MIN_CANDIDATES = 30
STAGE1_MAX_CANDIDATES = 120

for d in (DATA_DIR, CACHE_DIR, LABELED_DIR, SAMPLES_DIR):
    d.mkdir(parents=True, exist_ok=True)
