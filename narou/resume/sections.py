from __future__ import annotations

import re

SECTION_NAMES: dict[str, list[str]] = {
    "summary": ["summary", "profile", "objective", "about", "overview", "professional summary"],
    "experience": [
        "experience", "work experience", "employment", "professional experience",
        "work history", "career history", "relevant experience",
    ],
    "education": ["education", "academic background", "academics", "academic qualifications"],
    "skills": [
        "skills", "technical skills", "core skills", "competencies", "core competencies",
        "key skills", "certified competencies", "technologies", "technical competencies",
    ],
    "projects": ["projects", "selected projects", "personal projects", "research projects"],
    "certifications": ["certifications", "certification", "licenses", "certs", "credentials"],
    "publications": ["publications", "papers", "research", "publications and papers"],
    "awards": ["awards", "honors", "awards and honors", "recognition"],
}

_HEADER_RE = re.compile(
    r"^(?P<title>[A-Z][A-Z0-9 &/,\-]{2,60})\s*:?\s*$"
)
_EMAIL_RE = re.compile(r"[\w\.\+\-]+@[\w\-]+\.[\w\.\-]+")
_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s\.\-]?)?\(?\d{3}\)?[\s\.\-]?\d{3}[\s\.\-]?\d{4}"
)
_LINKEDIN_RE = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/in/[\w\-\_%/]+", re.I)
_GITHUB_RE = re.compile(r"(?:https?://)?(?:www\.)?github\.com/[\w\-\_/]+", re.I)
_URL_RE = re.compile(r"https?://[\w\-\.\/_%#?=&]+")


def _canonical(title: str) -> str | None:
    t = title.strip().lower().rstrip(":").strip()
    for key, aliases in SECTION_NAMES.items():
        if t in aliases:
            return key
    for key, aliases in SECTION_NAMES.items():
        for alias in aliases:
            if alias in t and len(t) <= len(alias) + 6:
                return key
    return None


def extract_sections(text: str) -> dict[str, str]:
    if not text:
        return {}
    lines = text.splitlines()
    sections: dict[str, list[str]] = {}
    current = "_preamble"
    sections[current] = []
    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            sections[current].append("")
            continue
        stripped = line.strip()
        canon = None
        if stripped.isupper() and 2 < len(stripped) < 60:
            canon = _canonical(stripped)
        if canon is None:
            m = _HEADER_RE.match(stripped)
            if m:
                canon = _canonical(m.group("title"))
        if canon is not None:
            current = canon
            sections.setdefault(current, [])
            continue
        sections[current].append(line)

    out: dict[str, str] = {}
    for key, chunks in sections.items():
        joined = "\n".join(chunks).strip()
        if joined:
            out[key] = joined

    if "summary" not in out and "_preamble" in out:
        preamble_lines = out["_preamble"].splitlines()
        body_lines = [
            ln for ln in preamble_lines
            if not _EMAIL_RE.search(ln)
            and not _PHONE_RE.search(ln)
            and not _URL_RE.search(ln)
            and len(ln.split()) > 3
        ]
        if body_lines:
            out["summary"] = "\n".join(body_lines).strip()
    return out


def extract_contact(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    em = _EMAIL_RE.search(text)
    if em:
        out["email"] = em.group(0)
    ph = _PHONE_RE.search(text)
    if ph:
        normalized = re.sub(r"[^\d+]", "", ph.group(0))
        if 10 <= len(normalized) <= 15:
            out["phone"] = ph.group(0).strip()
    li = _LINKEDIN_RE.search(text)
    if li:
        out["linkedin"] = li.group(0)
    gh = _GITHUB_RE.search(text)
    if gh:
        out["github"] = gh.group(0)

    first_lines = [ln.strip() for ln in text.splitlines()[:8] if ln.strip()]
    for line in first_lines:
        if "@" in line or "http" in line.lower():
            continue
        if re.search(r"\d{3}", line):
            continue
        if 3 < len(line) < 60 and all(
            tok[0].isupper() for tok in line.split()[:4] if tok and tok[0].isalpha()
        ):
            words = line.split()
            if 1 < len(words) <= 5 and all(w[0].isalpha() for w in words if w):
                out["name"] = line
                break
    return out


_SKILL_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\+\#\.\-]{1,}(?:\s+[A-Za-z][A-Za-z0-9\+\#\.\-]+)?")
_SKILL_STOP = {
    "and", "or", "the", "with", "for", "experience", "years", "knowledge",
    "understanding", "ability", "skills", "proficient", "strong", "excellent",
}


def _strip_category_header(text: str) -> str:
    """Strip category prefixes like 'Cloud Security: AWS' -> 'AWS'.

    Catches any pattern where a short label precedes a colon at the start
    of a skill chunk. Only triggers when the part after the colon looks
    like an actual skill (not empty, not too short).
    """
    idx = text.find(":")
    if idx < 0 or idx > 45:
        return text
    after = text[idx + 1:].strip(" .:\t-")
    if not after or len(after) < 2:
        return text
    before = text[:idx].strip()
    # Only strip if the prefix looks like a category label (mostly letters,
    # spaces, commas, ampersands -- not a skill like "C++:" or "Node.js:")
    if len(before) < 4:
        return text
    if re.match(r"^[A-Za-z][A-Za-z\s,&/\-]+$", before):
        return after
    return text


def extract_skills(sections: dict[str, str]) -> list[str]:
    skills_text = sections.get("skills", "")
    if not skills_text:
        return []
    raw = re.split(r"[,\n;\|\u2022\u2023\u25E6\u2043\u00b7]|\s\-\s", skills_text)
    out: list[str] = []
    seen: set[str] = set()
    for chunk in raw:
        s = chunk.strip(" .:\t-")
        if not s or len(s) > 50:
            continue
        s = _strip_category_header(s)
        if not s or len(s) > 50:
            continue
        low = s.lower()
        if low in _SKILL_STOP:
            continue
        if low in seen:
            continue
        seen.add(low)
        out.append(s)
    return out
