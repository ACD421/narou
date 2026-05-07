# Narou: AI Job Search Assistant

Smart matching. Real opportunities. No more fake jobs.

**HCAI 4350 - Team Narou** - Andrew Dorman, Hugo Olivarez, Kristopher Norton, Kassel Williams, Jonathan Witte

## What it does

Narou compares an uploaded resume against live job postings (Greenhouse, Lever) and returns:

1. **Ranked matches** using a section-aware semantic similarity engine (SGM, 3.6 MB, non-neural, zero API cost)
2. **Ghost-job confidence** per posting via a heuristic + ML classifier trained on real/fake job data
3. **Company trust grades** (A-F) aggregated across all of a company's live postings
4. **Resume suggestions** generated from the gap between your resume and your top-matched roles
5. **System metrics**: parse accuracy, feed uptime, time-to-first-result, fraud flag rate

Everything runs locally. No API keys required.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload a resume (PDF or DOCX), wait for the corpus to seed, click Analyze. The app crawls ~1,200 Greenhouse and Lever boards automatically on first launch.

## Architecture

```
narou/
  ingestion/   Greenhouse + Lever adapters, canonical Job schema
  resume/      PDF/DOCX parser with section detection
  matching/    SGM wrapper + two-stage retrieval (fast filter -> SGM rerank)
  fraud/       Feature extraction + heuristics + logistic/random-forest classifier
  suggestions/ Template-based resume gap analysis
  storage/     SQLite cache for jobs, embeddings, analyses
models/sgm/    Pre-trained SGM engine (ng3 + ng23 + ng23big + ng34, 20 features)
app.py         Streamlit UI
```

## The matching engine

Narou uses SGM (Spectral Geometric Matching), a non-neural similarity engine that combines 12 trained character n-gram features (cosine + IDF-cosine + L2 similarity across 4 tokenization groups) with 7 lexical features and a synonym bridge. Total size: 3.6 MB. On STS-B it scores 0.76 Spearman vs MiniLM's 0.82, at 22x smaller and zero dependencies on PyTorch or sentence-transformers.

For resume-to-job scoring, narou performs two-stage retrieval:
1. **Stage 1**: Fast stem/char3 overlap filter ranks N candidates to top-K
2. **Stage 2**: Full SGM predict on top-K with section-weighted aggregation

## Fraud detection

Per the pipeline spec, the ghost-job classifier uses three feature families:

- **TEXT**: length, specificity (TF-IDF deviation), keyword density, buzzword ratio, readability
- **BEHAVIOR**: days active, repost count, near-duplicate similarity
- **COMPANY**: posting volume, posting velocity, title repeat rate

Models: logistic regression + random forest, trained on the public real/fake job postings dataset, evaluated on held-out test.

## Development

```bash
pytest tests/
python scripts/train_fraud_classifier.py
```
