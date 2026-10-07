"""FastAPI service for the RAG pipeline.

Run:  uvicorn api:app --reload
Docs: http://localhost:8000/docs

Environment variables:
    OPENAI_API_KEY    key for the chat model
    LLM_MODEL         chat model name (default gpt-3.5-turbo)
    RETRIEVAL_MODE    dense | hybrid | rerank (default hybrid)
    VECTORSTORE_PATH  Chroma folder (default data/vectorstore/chroma_db)
    API_KEY           if set, clients must send it in the X-API-Key header
"""
import logging
import os
import secrets
import time
from functools import lru_cache
from typing import List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
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
    version="1.1.0",
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
        retrieval_mode=os.getenv("RETRIEVAL_MODE", "hybrid"),
    )


def pipeline_dep():
    try:
        return get_pipeline()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    """Enforce the X-API-Key header only when API_KEY is configured."""
    expected = os.getenv("API_KEY")
    if not expected:
        return
    if not secrets.compare_digest((x_api_key or "").encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
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
