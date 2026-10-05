# RAG Document Q&A Pipeline

[![CI](https://github.com/SPpavani/RAG-Document-QA-Pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/SPpavani/RAG-Document-QA-Pipeline/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)

Ask questions about your own PDFs and get answers **with citations (file + page)**, served through a **FastAPI** service and backed by a **measurable retrieval evaluation**.

Built with LangChain, ChromaDB, HuggingFace embeddings, an OpenAI chat model, FastAPI and Docker.

## Architecture

```
PDFs ──► split (1000 / 200 overlap) ──► embed (MiniLM) ──► Chroma vector store
                                                                 │
POST /query ──► validate ──► retrieve top-k ──► grounded prompt ──► LLM (temp 0)
                                                                 │
                               { answer, sources[file, page, snippet], latency_ms }
```

## What's in the repo

| File | Purpose |
|---|---|
| `ingest.py` | Ingests **all PDFs in a folder** into Chroma. Rebuilds the index each run, so no duplicate chunks |
| `rag_pipeline.py` | `RAGPipeline`: retrieval + grounded generation. Temperature 0, "I don't know" fallback, citations with file and page |
| `api.py` | FastAPI app: `GET /health`, `POST /query` with input validation and clear error codes (422 / 503 / 500) |
| `evaluate.py` + `eval/questions.json` | Retrieval evaluation: hit-rate@k, MRR, latency p50/p95. Needs no LLM or API key |
| `gen_pdfs.py` | Generates a synthetic sample PDF so anyone can run the project end to end |
| `tests/` | Pytest suite for the API and evaluation metrics (no ML models needed) |
| `Dockerfile` | Container image for the API |
| `.github/workflows/ci.yml` | Runs the tests on every push and pull request |

## Quick start

```bash
git clone https://github.com/SPpavani/RAG-Document-QA-Pipeline.git
cd RAG-Document-QA-Pipeline
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env          # then add your OPENAI_API_KEY

python gen_pdfs.py            # sample PDF (or put your own PDFs in data/raw/pdfs/)
python ingest.py              # build the vector store
python evaluate.py            # measure retrieval quality
uvicorn api:app --reload      # start the API, docs at http://localhost:8000/docs
```

## API

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How much will Acme Corp invest in AI in 2024?"}'
```

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
| 422 | Question missing, shorter than 3 or longer than 500 characters |
| 503 | Vector store not found, so run `python ingest.py` |
| 500 | Pipeline error (details in server logs) |

## Docker

```bash
docker build -t rag-api .
docker run -p 8000:8000 --env-file .env -v "$(pwd)/data:/app/data" rag-api
```

## Evaluation

`python evaluate.py` runs the questions in `eval/questions.json` against the vector store and reports:

- **hit-rate@k**: share of questions where a retrieved chunk contains the expected facts
- **MRR**: how high the first relevant chunk ranks
- **latency p50 / p95**: retrieval speed

A chunk counts as relevant when it contains all the expected keywords for the question. This is a cheap, deterministic proxy for retrieval quality. It does not judge the final answer, which is why answer-level evaluation is on the roadmap.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Design decisions

- **Temperature 0**: factual Q&A should be repeatable, not creative.
- **Grounded prompt**: the model must say "I don't know" when the context lacks the answer.
- **Citations with file and page**: users can verify every answer.
- **Local embeddings (MiniLM)**: no embedding API cost, runs offline.
- **Lazy imports in the API**: tests run in CI in seconds without downloading ML models.

## Known limitations

- PDF text only (no OCR for scanned documents, tables are not parsed specially)
- Single-turn questions (no conversation memory)
- Dense retrieval only (no keyword/BM25 component)
- No authentication or rate limiting on the API

## Roadmap

- [ ] Hybrid search (BM25 + vectors) with a cross-encoder reranker, compared against this baseline
- [ ] Answer-level evaluation (faithfulness, answer relevance) with RAGAS
- [ ] Streaming responses and conversation memory
- [ ] PII redaction and prompt-injection guardrails
- [ ] Tracing and cost tracking (Langfuse / OpenTelemetry)
- [ ] API-key auth and rate limiting
- [ ] Deploy to Azure Container Apps or Hugging Face Spaces
