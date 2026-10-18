"""Measure retrieval quality. Usage:
    python -m scripts.eval                      # default MiniLM
    MODEL_NAME=BAAI/bge-small-en-v1.5 python -m scripts.eval
    RERANKER_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2 python -m scripts.eval
A question counts as a hit if any top-k result's parent_id is in `expected`.
"""
import json
import os
from pathlib import Path

from app.retriever import PolicyRetriever

ROOT = Path(__file__).resolve().parent.parent
MODEL = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER = os.getenv("RERANKER_NAME") or None

r = PolicyRetriever(ROOT / "data" / "corpus.jsonl", model_name=MODEL, cache_dir=ROOT / ".cache", reranker_name=RERANKER)
qs = json.loads((ROOT / "tests" / "eval_questions.json").read_text())

hits = {1: 0, 3: 0, 5: 0}
mrr = 0.0
misses = []
for item in qs:
    res = r.search(item["q"], top_k=5)
    parents = [x["parent_id"] for x in res]
    rank = next((i + 1 for i, p in enumerate(parents) if p in item["expected"]), None)
    for k in hits:
        if rank and rank <= k:
            hits[k] += 1
    mrr += 1 / rank if rank else 0
    if not rank or rank > 1:
        misses.append((item["q"], item["expected"], parents[:3], rank))

n = len(qs)
print(f"model={MODEL} reranker={RERANKER} questions={n}")
print(f"hit@1={hits[1]/n:.2f}  hit@3={hits[3]/n:.2f}  hit@5={hits[5]/n:.2f}  MRR={mrr/n:.3f}")
for q, exp, got, rank in misses:
    print(f"  rank={rank}  expected={exp}  got={got}  :: {q}")
