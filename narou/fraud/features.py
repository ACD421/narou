from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field

from ..schema import Job


BUZZWORDS = {
    "rockstar", "ninja", "guru", "wizard", "unicorn", "superhero", "jedi",
    "passionate", "passionately", "exciting", "fast-paced", "fast paced",
    "dynamic environment", "family", "work hard play hard", "hustle",
    "world-class", "world class", "industry-leading", "cutting-edge",
    "cutting edge", "best-in-class", "synergy", "game-changing",
    "move fast", "disruptive", "disrupting", "mission-driven", "wear many hats",
    "value-add", "results-oriented",
}

TECH_KEYWORDS = {
    "python", "java", "javascript", "typescript", "go", "golang", "rust",
    "c++", "c#", "ruby", "php", "kotlin", "swift", "scala", "r",
    "sql", "postgres", "postgresql", "mysql", "mongodb", "redis", "cassandra",
    "aws", "azure", "gcp", "kubernetes", "docker", "terraform", "ansible",
    "react", "vue", "angular", "svelte", "next.js", "nextjs", "django", "flask",
    "fastapi", "spring", "rails", "node", "nodejs", "express",
    "pytorch", "tensorflow", "sklearn", "scikit-learn", "hugging", "langchain",
    "kafka", "spark", "airflow", "snowflake", "databricks", "dbt",
    "ci/cd", "cicd", "jenkins", "git", "github", "gitlab", "linux",
    "siem", "soc", "soar", "edr", "xdr", "splunk", "elastic", "crowdstrike",
    "owasp", "nist", "iso 27001", "iso 42001", "soc 2", "pci", "pci-dss",
    "hipaa", "gdpr", "fedramp", "cmmc", "csa", "ccpa",
    "mitre att&ck", "mitre atlas", "cyber kill chain",
    "zero trust", "sase", "cspm", "cwpp", "casb",
    "penetration testing", "incident response", "threat intelligence",
    "vulnerability management", "red team", "blue team", "purple team",
    "reverse engineering", "malware analysis", "threat hunting",
    "detection engineering", "digital forensics", "dfir",
    "ai red teaming", "prompt injection", "ai security",
    "iam", "mfa", "sso", "rbac", "active directory",
    "figma", "sketch", "adobe", "tailwind", "sass", "css", "html",
    "bachelor", "master", "phd", "degree", "years of experience",
}

SALARY_RE = re.compile(
    r"\$\s?\d{2,3}[,\.]?\d{3}|\d{2,3}\s?(?:k|K)\b|compensation|salary range"
)
YEARS_RE = re.compile(r"\b(\d+)\s*\+?\s*(?:years?|yrs?)\b", re.I)
_SENT_SPLIT = re.compile(r"(?<=[\.\!\?])\s+")
_WORD_RE = re.compile(r"\b[a-zA-Z][a-zA-Z\-\+\#]+\b")


@dataclass
class FraudFeatures:
    job_uid: str

    word_count: int = 0
    char_count: int = 0
    sentence_count: int = 0
    avg_sentence_length: float = 0.0
    unique_ratio: float = 0.0
    caps_ratio: float = 0.0
    buzzword_ratio: float = 0.0
    tech_density: float = 0.0
    has_salary: int = 0
    has_years: int = 0
    specificity: float = 0.0
    url_count: int = 0
    exclamation_density: float = 0.0

    days_active: float = 0.0
    repost_count: int = 0
    dup_similarity: float = 0.0

    company_posting_volume: int = 0
    company_title_repeat_rate: float = 0.0
    company_posting_velocity: float = 0.0

    extra: dict = field(default_factory=dict)

    def to_vector(self) -> list[float]:
        return [
            float(self.word_count),
            float(self.char_count),
            float(self.sentence_count),
            self.avg_sentence_length,
            self.unique_ratio,
            self.caps_ratio,
            self.buzzword_ratio,
            self.tech_density,
            float(self.has_salary),
            float(self.has_years),
            self.specificity,
            float(self.url_count),
            self.exclamation_density,
            self.days_active,
            float(self.repost_count),
            self.dup_similarity,
            float(self.company_posting_volume),
            self.company_title_repeat_rate,
            self.company_posting_velocity,
        ]

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "word_count", "char_count", "sentence_count", "avg_sentence_length",
            "unique_ratio", "caps_ratio", "buzzword_ratio", "tech_density",
            "has_salary", "has_years", "specificity", "url_count",
            "exclamation_density", "days_active", "repost_count", "dup_similarity",
            "company_posting_volume", "company_title_repeat_rate", "company_posting_velocity",
        ]

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("extra", None)
        return d


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def _company_aggregates(jobs: list[Job]) -> dict[str, dict]:
    by_company: dict[str, list[Job]] = {}
    for j in jobs:
        by_company.setdefault(j.company, []).append(j)

    agg: dict[str, dict] = {}
    for company, cjobs in by_company.items():
        titles = [j.title.strip().lower() for j in cjobs]
        title_counts = Counter(titles)
        repeat = sum(1 for t, c in title_counts.items() if c > 1)
        title_repeat_rate = repeat / len(titles) if titles else 0.0

        days_list = [j.days_active for j in cjobs if j.days_active is not None]
        if days_list:
            max_days = max(days_list)
            velocity = len(cjobs) / max(1, max_days + 1)
        else:
            velocity = 0.0

        desc_hashes: dict[str, int] = {}
        for j in cjobs:
            desc_hashes[j.description_hash] = desc_hashes.get(j.description_hash, 0) + 1
        dup_by_hash = desc_hashes

        title_by_company: dict[str, int] = {}
        for t in titles:
            title_by_company[t] = title_by_company.get(t, 0) + 1

        agg[company] = {
            "jobs": cjobs,
            "posting_volume": len(cjobs),
            "title_repeat_rate": title_repeat_rate,
            "posting_velocity": velocity,
            "titles": title_by_company,
            "dup_hashes": dup_by_hash,
        }
    return agg


def _compute_idf(jobs: list[Job]) -> dict[str, float]:
    if not jobs:
        return {}
    df: dict[str, int] = {}
    for j in jobs:
        seen = set(_tokenize(j.description))
        for token in seen:
            df[token] = df.get(token, 0) + 1
    n = len(jobs)
    return {tok: math.log((n + 1) / (c + 1)) + 1.0 for tok, c in df.items()}


def _specificity(text: str, idf: dict[str, float]) -> float:
    if not text or not idf:
        return 0.0
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    scores = [idf.get(tok, 0.0) for tok in tokens]
    vals = [s for s in scores if s > 0]
    if not vals:
        return 0.0
    return sum(vals) / len(vals) / 8.0


def extract_features(
    job: Job,
    company_agg: dict[str, dict] | None = None,
    idf: dict[str, float] | None = None,
    dedup_map: dict[str, dict] | None = None,
) -> FraudFeatures:
    text = job.description or ""
    title = job.title or ""
    feats = FraudFeatures(job_uid=job.uid)

    words = _tokenize(text)
    sentences = [s for s in _SENT_SPLIT.split(text) if s.strip()]
    feats.word_count = len(words)
    feats.char_count = len(text)
    feats.sentence_count = len(sentences)
    feats.avg_sentence_length = (
        sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0.0
    )
    feats.unique_ratio = len(set(words)) / len(words) if words else 0.0

    uppers = sum(1 for c in text if c.isupper())
    letters = sum(1 for c in text if c.isalpha())
    feats.caps_ratio = uppers / letters if letters else 0.0

    if words:
        blob = " ".join(words)
        bz_hits = sum(1 for bz in BUZZWORDS if bz in blob)
        feats.buzzword_ratio = bz_hits / max(1, len(sentences))
        tech_hits = sum(1 for t in TECH_KEYWORDS if t in blob)
        feats.tech_density = tech_hits / max(1, len(words) / 100)

    feats.has_salary = int(bool(SALARY_RE.search(text)))
    feats.has_years = int(bool(YEARS_RE.search(text)))
    feats.url_count = text.count("http://") + text.count("https://")
    feats.exclamation_density = text.count("!") / max(1, len(sentences))

    if idf:
        feats.specificity = _specificity(text, idf)

    feats.days_active = float(job.days_active or 0)

    if company_agg is not None and job.company in company_agg:
        agg = company_agg[job.company]
        feats.company_posting_volume = agg["posting_volume"]
        feats.company_title_repeat_rate = agg["title_repeat_rate"]
        feats.company_posting_velocity = agg["posting_velocity"]

        title_lower = title.strip().lower()
        feats.repost_count = max(0, agg["titles"].get(title_lower, 1) - 1)

        hash_count = agg["dup_hashes"].get(job.description_hash, 1)
        feats.dup_similarity = 1.0 if hash_count > 1 else 0.0

    # Cross-company reposts: cluster_size counts identical descriptions found
    # anywhere in the corpus. This catches candidate-harvester reposts that
    # within-company aggregates miss.
    if dedup_map is not None:
        dinfo = dedup_map.get(job.uid)
        if dinfo:
            cluster_size = int(dinfo.get("cluster_size") or 1)
            if cluster_size > 1:
                feats.repost_count = max(feats.repost_count, cluster_size - 1)
                feats.dup_similarity = max(feats.dup_similarity, 1.0)
                feats.extra["cross_company_cluster_size"] = cluster_size

    return feats


def extract_batch(
    jobs: list[Job],
    dedup_map: dict[str, dict] | None = None,
    idf: dict[str, float] | None = None,
) -> list[FraudFeatures]:
    if not jobs:
        return []
    agg = _company_aggregates(jobs)
    # Per-query IDF on a tiny batch (e.g. top-20 matches) is both slow and
    # statistically meaningless. Skip unless the caller provides a corpus-
    # wide IDF or the batch is large enough to matter.
    if idf is None and len(jobs) >= 200:
        idf = _compute_idf(jobs)
    return [
        extract_features(j, company_agg=agg, idf=idf, dedup_map=dedup_map)
        for j in jobs
    ]
