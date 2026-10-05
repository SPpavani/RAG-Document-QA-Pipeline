"""FastAPI service for the RAG pipeline.

Run:  uvicorn api:app --reload
Docs: http://localhost:8000/docs
"""
import logging
import os
import time
from functools import lru_cache
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

try:  # optional: load variables from a local .env file
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger("rag_api")
VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "data/vectorstore/chroma_db")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class Source(BaseModel):
    content: str
    source: Optional[str] = None
    page: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    source_count: int
    latency_ms: float


app = FastAPI(
    title="RAG Document Q&A API",
    version="1.0.0",
    description="Ask questions about ingested PDFs and get answers with citations.",
)


@lru_cache(maxsize=1)
def get_pipeline():
    """Build the pipeline once. Heavy imports happen lazily so tests stay light."""
    if not os.path.isdir(VECTORSTORE_PATH):
        raise FileNotFoundError(
            f"Vector store not found at '{VECTORSTORE_PATH}'. Run `python ingest.py` first."
        )
    from rag_pipeline import RAGPipeline

    return RAGPipeline(
        vectorstore_path=VECTORSTORE_PATH,
        llm_model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


def pipeline_dep():
    try:
        return get_pipeline()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, pipeline=Depends(pipeline_dep)):
    start = time.perf_counter()
    try:
        result = pipeline.query(req.question)
    except Exception:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail="Failed to process the question.")
    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    logger.info("query ok | %.1f ms | %d sources", latency_ms, result["source_count"])
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        source_count=result["source_count"],
        latency_ms=latency_ms,
    )
