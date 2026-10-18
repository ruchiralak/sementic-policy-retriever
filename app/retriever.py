"""Hybrid policy retriever: dense (sentence-transformers) + BM25, fused with RRF."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

# keeps course codes (EEI4362) and numbers (3.30, 40) as single tokens
TOKEN_RE = re.compile(r"[A-Za-z]+\d+|\d+(?:\.\d+)?|[A-Za-z]+")
STOP = {"the", "a", "an", "of", "to", "in", "and", "or", "is", "are", "for", "on", "be", "do", "i", "my", "can", "what", "how"}

# audience -> which `applies_to` tags a chunk must carry
AUDIENCE_TAGS = {
    "applicant": {"applicants"},
    "new_student": {"new_students", "all_students"},
    "student": {"all_students", "new_students"},
}


def tokenize(text: str) -> list[str]:
    return [t for t in (m.lower() for m in TOKEN_RE.findall(text)) if t not in STOP]


class PolicyRetriever:
    def __init__(
        self,
        corpus_path: str | Path,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: str | Path = ".cache",
        reranker_name: str | None = None,
        rrf_k: int = 60,
    ):
        self.model_name = model_name
        self.rrf_k = rrf_k
        self.docs = [json.loads(l) for l in Path(corpus_path).read_text(encoding="utf-8").splitlines() if l.strip()]
        self.model = SentenceTransformer(model_name)
        self.embeddings = self._load_or_build_embeddings(Path(cache_dir), Path(corpus_path))
        self.bm25 = BM25Okapi([tokenize(d["embed_text"] + " " + " ".join(d.get("keywords", []))) for d in self.docs])
        self.reranker = CrossEncoder(reranker_name) if reranker_name else None

    # ---- indexing -------------------------------------------------------
    def _load_or_build_embeddings(self, cache_dir: Path, corpus_path: Path) -> np.ndarray:
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha1((self.model_name + corpus_path.read_text(encoding="utf-8")).encode()).hexdigest()[:16]
        f = cache_dir / f"emb_{key}.npy"
        if f.exists():
            return np.load(f)
        emb = self.model.encode([d["embed_text"] for d in self.docs], normalize_embeddings=True, show_progress_bar=False)
        np.save(f, emb)
        return emb

    # ---- search ---------------------------------------------------------
    def _mask(self, audience: str | None, topic: str | None) -> np.ndarray:
        ok = np.ones(len(self.docs), dtype=bool)
        for i, d in enumerate(self.docs):
            if audience:
                tags = AUDIENCE_TAGS.get(audience, set())
                if not tags & set(d.get("applies_to", [])):
                    ok[i] = False
            if topic and d.get("topic") != topic:
                ok[i] = False
        return ok

    def search(self, query: str, top_k: int = 5, audience: str | None = None, topic: str | None = None, pool: int = 20) -> list[dict]:
        mask = self._mask(audience, topic)
        if not mask.any():
            return []

        q = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        dense = self.embeddings @ q
        sparse = self.bm25.get_scores(tokenize(query))

        dense = np.where(mask, dense, -np.inf)
        sparse = np.where(mask, sparse, -np.inf)
        d_rank = {int(i): r for r, i in enumerate(np.argsort(-dense)[:pool])}
        s_order = [int(i) for i in np.argsort(-sparse)[:pool] if sparse[i] > 0]
        s_rank = {i: r for r, i in enumerate(s_order)}

        fused: dict[int, float] = {}
        for rank_map in (d_rank, s_rank):
            for i, r in rank_map.items():
                if mask[i]:
                    fused[i] = fused.get(i, 0.0) + 1.0 / (self.rrf_k + r + 1)
        ranked = sorted(fused, key=fused.get, reverse=True)[:pool]

        if self.reranker and ranked:
            scores = self.reranker.predict([(query, self.docs[i]["embed_text"]) for i in ranked])
            ranked = [i for _, i in sorted(zip(scores, ranked), reverse=True)]
            final = {i: float(s) for s, i in zip(sorted(scores, reverse=True), ranked)}
        else:
            final = fused

        out = []
        for i in ranked[:top_k]:
            d = self.docs[i]
            out.append({
                "id": d["id"],
                "parent_id": d["parent_id"],
                "section": d["section"],
                "topic": d["topic"],
                "page": d["page"],
                "applies_to": d["applies_to"],
                "text": d["text"],
                "score": round(float(final[i]), 5),
                "dense_score": round(float(dense[i]), 4),
                "keyword_match": i in s_rank,
            })
        return out

    def topics(self) -> list[str]:
        return sorted({d["topic"] for d in self.docs})
