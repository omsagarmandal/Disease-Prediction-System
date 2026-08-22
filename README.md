# Project 01 — News AI Application (AI/ML Module)

**Intern:** Omsagar
**Role:** AI/ML Intern (ai_model_dev_branch scope)
**Status:** Core pipeline complete
**Started:** 2026-07-15

## Objective
Build the AI/ML pipeline for the News AI Application — ingest news articles from multiple RSS sources, enrich them with AI (sentiment, summary, topic classification), generate embeddings, cluster related articles, and fine-tune a sentiment model for improved accuracy.

## Scope
This module covers AI/ML tasks only:
- RSS ingestion + deduplication
- Article text extraction
- AI enrichment (sentiment, summary, topic)
- Embeddings + semantic clustering
- Model fine-tuning + evaluation
- Cost/token usage tracking

Frontend, backend CMS, and UI/UX are out of scope (handled by other branches: frontend_dev_branch, backend_dev_branch, ui_ux_branch).

## Tech Stack
- Python (venv)
- PostgreSQL + pgvector (Docker)
- Redis (Docker)
- transformers (<5.0.0), torch, sentence-transformers
- Models: distilbert (sentiment), distilbart-cnn (summary), bart-large-mnli (topic, zero-shot)

## Setup
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
docker compose up -d

## Project Layout
project-01/
├── ai-service/
│   ├── enrich.py              # sentiment + summary + topic + usage logging
│   ├── generate_embeddings.py # sentence embeddings (384-dim)
│   ├── cluster_articles.py    # cosine similarity clustering
│   ├── finetune_sentiment.py  # fine-tuning on IMDB dataset
│   └── compare_models.py      # pretrained vs fine-tuned comparison
├── workers/
│   ├── fetch_rss.py           # RSS ingestion + dedup
│   └── extract_articles.py    # trafilatura text extraction
├── db/init.sql                # 7 tables (incl. ai_usage_log)
├── docker-compose.yml
└── .env

## Progress
- [x] Multiple RSS sources (7 sources: BBC, CNN, Al Jazeera, Guardian, NYT, NPR, Sky News)
- [x] Article extraction (trafilatura + fallback logic)
- [x] AI enrichment on all articles (sentiment, summary, topic)
- [x] Embeddings + clustering on all articles
- [x] Model fine-tuning (sentiment)
- [x] Evaluation metrics (accuracy, F1, confusion matrix, pretrained vs fine-tuned comparison)
- [x] Cost/token tracking (ai_usage_log table: model_ver, tokens, latency)
- [ ] Mentor review / PR opened

## Results

| Item | Value |
|---|---|
| Total articles fetched | 241 |
| Articles enriched | 196 |
| Embeddings generated | 196 |
| Clusters formed | 14 |
| Pretrained model accuracy | 0.84 (F1 0.826) |
| Fine-tuned model accuracy | 0.81 (F1 0.817) |

Note: Fine-tuned model showed slightly lower accuracy than the base pretrained model, likely due to limited training data (500 samples, 1 epoch) — a valid finding, documented honestly rather than reported as an unqualified improvement.

## How to Run
python workers/fetch_rss.py
python workers/extract_articles.py
python ai-service/enrich.py
python ai-service/generate_embeddings.py
python ai-service/cluster_articles.py
python ai-service/finetune_sentiment.py
python ai-service/compare_models.py

## Links
- Personal repo: https://github.com/omsagarmandal/news-ai-app
- Team repo / PR: (to be added)