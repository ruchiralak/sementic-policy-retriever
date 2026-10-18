
# BSEHons Policy Retrieval (FastAPI tool)

Semantic + keyword search over the OUSL BSEHons 2025/2026 Student Guidebook. Built to be one tool in an agentic chatbot.

## How it works
- `data/source_corpus.jsonl` – 54 hand-structured policy passages (with section, page, topic, audience).
- `scripts/rechunk.py` – splits them into 96 smaller chunks (<= ~85 words) so nothing is truncated by MiniLM's 256-token limit. Re-run after editing the source.
- `app/retriever.py` – dense search (all-MiniLM-L6-v2) + BM25 (keeps course codes like `EEI4362` and numbers like `3.30` intact), fused with Reciprocal Rank Fusion. Optional cross-encoder reranker and metadata filters.
- `app/main.py` – FastAPI service.

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/rechunk.py            # only needed if you change the source corpus
uvicorn app.main:app --reload --port 8000
```
First start downloads the model (~90 MB) and caches embeddings in `.cache/`.

```bash
curl -s localhost:8000/search -H 'content-type: application/json' \
  -d '{"query":"What are the prerequisites for EEI5467?","top_k":3}'
```
Interactive docs: http://localhost:8000/docs

## Endpoints
| Method | Path | Purpose |
|---|---|---|
| POST | `/search` | `{query, top_k=5, audience?, topic?}` -> ranked passages with section, page, scores |
| GET | `/tool-spec` | Tool definition (name, description, JSON schema) to register in your agent |
| GET | `/topics` | Valid `topic` filter values |
| GET | `/health` | Model, reranker, chunk count |

`audience`: `applicant`, `new_student` or `student`.

## Config (env vars)
- `MODEL_NAME` (default `sentence-transformers/all-MiniLM-L6-v2`; try `BAAI/bge-small-en-v1.5`)
- `RERANKER_NAME` (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`; off by default)
- `CORPUS_PATH`

## Measure quality before trusting it
```bash
python -m scripts.eval
MODEL_NAME=BAAI/bge-small-en-v1.5 python -m scripts.eval
RERANKER_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2 python -m scripts.eval
```
`tests/eval_questions.json` has 33 student-style questions with the expected rule ids. Add more from real student questions; reports hit@1/3/5 and MRR and lists misses.

## Using it from the agent
Fetch `GET /tool-spec`, register it as a tool, and have the tool call `POST /search`. Instruct the agent to answer only from returned passages and cite `section` + `page`. If `results` is empty or scores are low, it should say the guidebook doesn't cover it and point to bseinfo@ousl.lk.

## Known limits
- Prerequisite eligibility ("can I register for X given what I passed?") needs rule logic over structured course data, not retrieval. Make that a separate tool.
- Dates and fees are for the 2025/2026 intake; regenerate the corpus each year.
- Source inconsistencies are noted inside the affected chunks (e.g. payment example 30 vs 31 credits).
=======

