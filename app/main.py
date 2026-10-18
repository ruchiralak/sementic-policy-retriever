"""FastAPI service exposing policy retrieval as a tool for an agentic chatbot."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.retriever import PolicyRetriever

ROOT = Path(__file__).resolve().parent.parent
CORPUS = os.getenv("CORPUS_PATH", str(ROOT / "data" / "corpus.jsonl"))
MODEL = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER = os.getenv("RERANKER_NAME") or None  # e.g. cross-encoder/ms-marco-MiniLM-L-6-v2

state: dict[str, PolicyRetriever] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["r"] = PolicyRetriever(CORPUS, model_name=MODEL, cache_dir=ROOT / ".cache", reranker_name=RERANKER)
    yield
    state.clear()


app = FastAPI(
    title="BSEHons Policy Retrieval",
    version="0.1.0",
    description="Semantic + keyword search over the OUSL BSEHons 2025/2026 student guidebook.",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "app" / "static" / "index.html")


class SearchRequest(BaseModel):
    model_config = {
        "json_schema_extra": {
            "examples": [{"query": "What are the prerequisites for EEI5467?", "top_k": 3}]
        }
    }

    query: str = Field(..., min_length=2, max_length=500, description="Student's question in natural language")
    top_k: int = Field(5, ge=1, le=10)
    audience: Literal["applicant", "new_student", "student"] | None = Field(
        None, description="Optional filter: who is asking"
    )
    topic: str | None = Field(None, description="Optional topic filter, see GET /topics")


class Passage(BaseModel):
    id: str
    parent_id: str
    section: str
    topic: str
    page: str
    applies_to: list[str]
    text: str
    score: float
    dense_score: float
    keyword_match: bool


class SearchResponse(BaseModel):
    query: str
    model: str
    results: list[Passage]


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL, "reranker": RERANKER, "chunks": len(state["r"].docs)}


@app.get("/topics")
def topics():
    return {"topics": state["r"].topics()}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    req.topic = (req.topic or "").strip() or None  # empty string means no filter
    if req.topic and req.topic not in state["r"].topics():
        raise HTTPException(status_code=422, detail=f"Unknown topic '{req.topic}'. See GET /topics.")
    results = state["r"].search(req.query, top_k=req.top_k, audience=req.audience, topic=req.topic)
    return SearchResponse(query=req.query, model=MODEL, results=results)


@app.get("/tool-spec")
def tool_spec():
    """Tool definition an agent framework can register (Anthropic / OpenAI-style JSON schema)."""
    return {
        "name": "search_policy",
        "description": (
            "Search the OUSL BSEHons student guidebook (2025/2026) for rules and policies. Use for questions about "
            "admission, registration, fees, course prerequisites, assessment, exam eligibility, grading, GPA, degree "
            "requirements, exemptions, industrial training and student services. Returns the most relevant guidebook "
            "passages with section and page; answer only from them and cite the section/page."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The student's question, rephrased as a self-contained search query"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                "audience": {"type": "string", "enum": ["applicant", "new_student", "student"], "description": "Optional"},
                "topic": {"type": "string", "description": "Optional topic filter from GET /topics"},
            },
            "required": ["query"],
        },
        "http": {"method": "POST", "path": "/search"},
    }