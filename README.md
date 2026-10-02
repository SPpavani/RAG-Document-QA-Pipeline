# RAG Document Q&A Pipeline

Ask questions about your own PDFs and get answers **with the source passages they came from**.
Built with LangChain, ChromaDB, HuggingFace embeddings and an OpenAI chat model.

## How it works

```
PDF ──► split into chunks ──► embed (MiniLM) ──► Chroma vector store
                                                        │
question ──► retrieve top-k chunks ──► prompt + LLM ──► answer + sources
```

| Step | File | Details |
|---|---|---|
| Ingest | `ingest.py` | Loads a PDF with `PyPDFLoader`, splits it (1000 chars, 200 overlap), embeds it and saves it to a persistent Chroma DB |
| Retrieve + generate | `rag_pipeline.py` | `RAGPipeline` class: loads the Chroma store, retrieves the top `k` chunks (default 3) and answers with a grounded prompt |
| Grounding | `rag_pipeline.py` | The prompt tells the model to say "I don't know" instead of inventing an answer |
| Output | `rag_pipeline.py` | Returns the answer, the source snippets and the source count |

**Models:** `sentence-transformers/all-MiniLM-L6-v2` for embeddings; `gpt-3.5-turbo` by default for generation (configurable).

## Quick start

```bash
git clone https://github.com/SPpavani/RAG-Document-QA-Pipeline.git
cd RAG-Document-QA-Pipeline
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"

# 1. put your PDF in data/raw/pdfs/ and point ingest.py at it
python ingest.py

# 2. ask questions
python query.py
```

## Example

```python
from rag_pipeline import RAGPipeline

rag = RAGPipeline(vectorstore_path="data/vectorstore/chroma_db", openai_api_key="...")
result = rag.query("What is the company's AI strategy?")
print(result["answer"])
print(result["sources"])
```

## Status and roadmap

Working prototype with a single-PDF ingestion path.

- [ ] Ingest a whole folder of PDFs
- [ ] Add a FastAPI endpoint and a small web UI
- [ ] Add an evaluation set (retrieval hit rate, answer faithfulness, latency)
- [ ] Add a reranker and compare against the current baseline
- [ ] Add unit tests and a GitHub Actions workflow
- [ ] Move the source files into a `src/` package
