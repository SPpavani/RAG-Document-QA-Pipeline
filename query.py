"""Ask questions from the command line.

Usage:
    python query.py "What is the AI strategy?"                    # answer + sources
    python query.py "What is the AI strategy?" --retrieval-only    # no LLM, no API key
    python query.py                                                # interactive mode
    python query.py "..." --mode rerank --k 5
"""
import argparse
import sys
from pathlib import Path

from config import Config
from retrieval import MODES


def _where(source, page) -> str:
    return f"{source or '?'} p.{page + 1 if isinstance(page, int) else '?'}"


def build_retriever(mode: str, k: int):
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    from retrieval import HybridRetriever

    embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
    db = Chroma(persist_directory=str(Config.VECTORSTORE_DIR), embedding_function=embeddings)
    return HybridRetriever(db, k=k, mode=mode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="omit for interactive mode")
    parser.add_argument("--mode", choices=MODES, default=Config.RETRIEVAL_MODE)
    parser.add_argument("--k", type=int, default=Config.SEARCH_K)
    parser.add_argument("--retrieval-only", action="store_true", help="skip the LLM")
    args = parser.parse_args()

    if not Path(Config.VECTORSTORE_DIR).is_dir():
        sys.exit("Vector store not found. Run `python ingest.py` first.")

    if args.retrieval_only:
        retriever = build_retriever(args.mode, args.k)

        def ask(question: str) -> None:
            for i, doc in enumerate(retriever.retrieve(question), start=1):
                meta = doc.metadata or {}
                print(f"\n[{i}] {_where(meta.get('source'), meta.get('page'))}")
                print(doc.page_content[:300])

    else:
        if not Config.OPENAI_API_KEY:
            sys.exit("OPENAI_API_KEY not set. Add it to .env, or use --retrieval-only.")
        from rag_pipeline import RAGPipeline

        pipeline = RAGPipeline(
            vectorstore_path=str(Config.VECTORSTORE_DIR),
            llm_model=Config.LLM_MODEL,
            search_k=args.k,
            openai_api_key=Config.OPENAI_API_KEY,
            retrieval_mode=args.mode,
        )

        def ask(question: str) -> None:
            result = pipeline.query(question)
            print(f"\nAnswer: {result['answer']}\n\nSources:")
            for i, s in enumerate(result["sources"], start=1):
                print(f"  [{i}] {_where(s['source'], s['page'] - 1 if s['page'] else None)}")

    if args.question:
        ask(args.question)
        return

    print("Type a question (or 'quit' to exit).")
    while True:
        try:
            question = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break
        ask(question)


if __name__ == "__main__":
    main()
