# RAG Document Q&A Pipeline

[![CI](https://github.com/SPpavani/RAG-Document-QA-Pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/SPpavani/RAG-Document-QA-Pipeline/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)

Ask questions about your own PDFs and get answers **with citations (file + page)**. Retrieval uses **hybrid search (BM25 + vectors) with an optional cross-encoder reranker**, is served through a **FastAPI** service, and is **evaluated side by side** against a plain vector-search baseline.

Built with LangChain, ChromaDB, HuggingFace embeddings, an OpenAI chat model, FastAPI and Docker.

## Architecture

```
PDFs ──► split (1000 / 200 overlap) ──► embed (MiniLM) ──► Chroma vector store
                                                                 │
                  ┌──────────────────────────────────────────────┤
                  ▼                                              ▼
        dense search (top 10)                          BM25 keyword search (top 10)
                  └───────────── Reciprocal Rank Fusion ─────────┘
                                         │
                       (optional) cross-encoder rerank
                                         │
POST /query ──► validate ──► top-k chunks ──► grounded prompt ──► LLM (temp 0)
                                         │
                   { answer, sources[file, page, snippet], latency_ms }
```

### Retrieval modes (`RETRIEVAL_MODE`)

| Mode | What it does | Good at |
|---|---|---|
| `dense` | Vector similarity only (the baseline) | Paraphrased questions |
| `hybrid` (default) | Vector + BM25, merged with Reciprocal Rank Fusion | Paraphrases **and** exact terms (names, codes, acronyms) |
| `rerank` | Hybrid shortlist, re-scored by a cross-encoder (`ms-marco-MiniLM-L-6-v2`) | Highest precision, slower |

## What's in the repo

| File | Purpose |
|---|---|
| `retrieval.py` | BM25 (written from scratch), Reciprocal Rank Fusion and `HybridRetriever` |
| `rag_pipeline.py` | `RAGPipeline`: retrieval + grounded generation. Temperature 0, "I don't know" fallback, citations with file and page |
| `api.py` | FastAPI app: `GET /health`, `POST /query`, optional API-key auth, clear error codes |
| `ingest.py` | Ingests **all PDFs in a folder** into Chroma. Rebuilds the index each run, so no duplicate chunks |
| `evaluate.py` + `eval/questions.json` | Compares dense / hybrid / rerank on hit-rate@k, MRR and latency. No LLM or API key needed |
| `gen_pdfs.py` | Generates a synthetic sample PDF so anyone can run the project end to end |
| `tests/` | Pytest suite covering the API, auth, retrieval logic and metrics (no ML models needed) |
| `Dockerfile`, `Makefile` | Container image and shortcut commands |
| `.github/workflows/ci.yml` | Syntax check + tests on every push and pull request |

## Quick start

```bash
git clone https://github.com/SPpavani/RAG-Document-QA-Pipeline.git
cd RAG-Document-QA-Pipeline
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env          # then add your OPENAI_API_KEY

python gen_pdfs.py            # sample PDF (or put your own PDFs in data/raw/pdfs/)
python ingest.py              # build the vector store
python evaluate.py            # compare dense vs hybrid vs rerank
uvicorn api:app --reload      # start the API, docs at http://localhost:8000/docs
```

Shortcuts: `make install`, `make sample`, `make ingest`, `make eval`, `make serve`, `make test`.

## API

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"question": "How much will Acme Corp invest in AI in 2024?"}'
```

(`X-API-Key` is only required if you set `API_KEY` in `.env`.)

Response shape:

```json
{
  "answer": "...",
  "sources": [{"content": "...", "source": "ai_strategy_2024.pdf", "page": 1}],
  "source_count": 3,
  "latency_ms": 812.4
}
```

| Status | Meaning |
|---|---|
| 200 | Answer returned |
| 401 | `API_KEY` is configured and the `X-API-Key` header is missing or wrong |
| 422 | Question missing, shorter than 3 or longer than 500 characters |
| 503 | Vector store not found, so run `python ingest.py` |
| 500 | Pipeline error (details in server logs) |

## Docker

```bash
docker build -t rag-api .
docker run -p 8000:8000 --env-file .env -v "$(pwd)/data:/app/data" rag-api
```

## Evaluation

`python evaluate.py` runs every question in `eval/questions.json` against each retrieval mode and prints a comparison table with:

- **hit-rate@k**: share of questions where a retrieved chunk contains the expected facts
- **MRR**: how high the first relevant chunk ranks
- **latency p50 / p95**: retrieval speed (the reranker's model-loading time is excluded by a warm-up call)

It also lists the questions each mode missed, and saves everything to `eval/results.json`.

A chunk counts as relevant when it contains all the expected keywords for the question. This is a cheap, deterministic proxy for retrieval quality. It does not judge the final answer, which is why answer-level evaluation is on the roadmap.

Use your own PDFs and questions for a meaningful comparison: the bundled 6-page synthetic document is small enough that every mode may score close to 100%.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Design decisions

- **Hybrid over dense-only**: embeddings blur exact terms such as part numbers and names; BM25 catches them. A unit test shows a case dense search misses and hybrid recovers.
- **Reciprocal Rank Fusion**: merges rankings without normalising incompatible score scales.
- **BM25 from scratch**: about 30 lines, fully unit-tested, no extra dependency.
- **Temperature 0 and a grounded prompt**: factual Q&A should be repeatable and say "I don't know" when the context lacks the answer.
- **Citations with file and page**: users can verify every answer.
- **Local embeddings (MiniLM)**: no embedding API cost, runs offline.
- **Lazy imports**: tests run in CI in seconds without downloading ML models.

## Known limitations

- PDF text only (no OCR for scanned documents, tables are not parsed specially)
- Single-turn questions (no conversation memory)
- The BM25 index is rebuilt in memory at startup, which suits thousands of chunks, not millions
- Small, synthetic evaluation set and a keyword-based relevance proxy
- No rate limiting on the API

## Roadmap

- [x] Hybrid search (BM25 + vectors) and cross-encoder reranker, compared by `evaluate.py`
- [x] API-key authentication
- [ ] Answer-level evaluation (faithfulness, answer relevance) with RAGAS
- [ ] Streaming responses and conversation memory
- [ ] PII redaction and prompt-injection guardrails
- [ ] Tracing and cost tracking (Langfuse / OpenTelemetry)
- [ ] Rate limiting
- [ ] Deploy to Azure Container Apps or Hugging Face Spaces
