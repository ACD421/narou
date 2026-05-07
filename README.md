# Narou: AI Job Search Assistant

Smart matching. Real opportunities. No more fake jobs.

**HCAI 4350 - Team Narou** - Andrew Dorman, Hugo Olivarez, Kristopher Norton, Kassel Williams, Jonathan Witte

**Live demo:** [teamnarou.streamlit.app](https://teamnarou.streamlit.app)

## What it does

Narou compares an uploaded resume against 53,000+ live job postings and returns:

1. **Ranked matches** using a section-aware semantic similarity engine (SGM, 3.6 MB, non-neural, zero API cost)
2. **Ghost-job detection** per posting via a heuristic + ML classifier trained on real/fake job data
3. **Freshness scoring** that penalizes stale postings and boosts recent ones in rankings
4. **Company trust grades** (A-F) aggregated across all of a company's postings in the corpus
5. **Resume gap analysis** with missing keywords, section advice, and rephrase hints
6. **System metrics**: corpus health, feed uptime, time-to-first-result, fraud flag rate

Everything runs locally. No API keys required. No resume data leaves the machine.

## Quick start

### Cloud (no install)

Visit [teamnarou.streamlit.app](https://teamnarou.streamlit.app), upload your resume, click Analyze.

### Local

```
# Windows
install.bat
run.bat

# macOS / Linux
chmod +x install.sh run.sh
./install.sh
./run.sh
```

Upload a resume (PDF or DOCX), wait for the index to build on first launch (~2 min), click Analyze.

## How it works

### Search pipeline

1. **Resume parsing**: PDF/DOCX text extraction with section detection (summary, experience, skills, education) and skill tokenization with automatic category-header stripping
2. **Stage-1 retrieval**: Dual TF-IDF vectorizers (char n-gram + word n-gram) score the entire 53K corpus via sparse matrix multiply -- sub-second
3. **Stage-2 rerank**: Top candidates get full SGM similarity scoring across five resume sections (title, skills, summary, experience, lexical) with synonym bridging
4. **Freshness factor**: Recent postings (<7d) get a 5% boost; stale postings (90d+) get a 20% penalty
5. **Fraud scoring**: Heuristic + ML classifier flags ghost jobs based on text features, posting behavior, and cross-company dedup
6. **Company grades**: Full corpus analysis of each matched company's posting volume, repost rate, median age, and fraud flag rate

### The matching engine (SGM)

SGM (Spectral Geometric Matching) is a non-neural similarity engine combining 12 trained character n-gram features (cosine + IDF-cosine + L2 across 4 tokenization groups) with 7 lexical features and a synonym bridge. Total size: 3.6 MB. On STS-B it scores 0.76 Spearman vs MiniLM's 0.82, at 22x smaller and zero PyTorch dependency.

SGM powers the stage-2 rerank. Stage-1 retrieval uses dual TF-IDF vectorizers (char 3-4 grams + word 1-2 grams) over the full corpus via sparse cosine similarity. This two-stage design searches 53K+ jobs in under a second: the TF-IDF pass is a single matrix multiply, and only the top candidates (default 150) go through the full SGM scoring pipeline.

### Fraud detection

The ghost-job classifier uses three feature families:

- **TEXT**: length, specificity (TF-IDF deviation), keyword density, buzzword ratio, readability
- **BEHAVIOR**: days active, repost count, near-duplicate similarity
- **COMPANY**: posting volume, posting velocity, title repeat rate

Models: logistic regression + random forest ensemble, trained on the public real/fake job postings dataset.

### Data sources

- **Greenhouse**: ~1,163 company job boards via public API (`boards-api.greenhouse.io`)
- **Lever**: ~35 company job boards via public API (`api.lever.co`)
- All data from public job postings. No scraping, no authentication required.

## Corpus management

The job corpus is pre-built and bundled with the project:

- **Seed database**: `data/jobs_seed.sqlite.gz` (53K+ jobs, compressed)
- **TF-IDF index**: Hosted on GitHub Release `v1.0-data` (downloaded on cloud, built locally)
- **Auto-refresh**: GitHub Action (`.github/workflows/refresh-corpus.yml`) crawls all boards Mon and Thu at 6am UTC, rebuilds the index, and commits fresh data
- **Local refresh**: Click "Refresh corpus now" in the sidebar to crawl live boards on demand

| Feature | Cloud | Local |
|---------|-------|-------|
| Corpus source | Pre-built seed from repo | Seed + live crawl |
| TF-IDF index | Downloaded from GitHub Release | Built on first launch (~2 min) |
| Refresh button | Hidden (auto-updates via CI) | Available in sidebar |
| Job loading | Lazy (on demand from SQLite) | Lazy (on demand from SQLite) |

## Architecture

```
narou/
  ingestion/   Greenhouse + Lever adapters, parallel crawler (24 threads)
  resume/      PDF/DOCX parser with section detection + skill extraction
  matching/    SGM engine + two-stage retrieval (TF-IDF -> SGM rerank)
  fraud/       Feature extraction + heuristics + logistic/RF classifier
  suggestions/ Keyword gap analysis + section advice + rephrase hints
  storage/     SQLite with thread-local connections + lazy job loading
models/sgm/    Pre-trained SGM engine (4 tokenization groups, 20 features)
app.py         Streamlit UI with conditional color coding
```

## Development

```bash
pip install -r requirements.txt
pytest tests/                              # 39 tests
python scripts/train_fraud_classifier.py   # retrain fraud model on fresh data
```

## Disclaimer

Narou is designed to assist job seekers in navigating job opportunities, not to make decisions on their behalf. Results are generated using automated matching and should be treated as guidance only. Fraud and ghost-job scores are statistical estimates, not guarantees. Final decisions should always depend on your own judgment and independent verification of each opportunity.
