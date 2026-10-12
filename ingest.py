"""Ingest every PDF in a folder into a persistent Chroma vector store.

Usage:
    python ingest.py [--pdf-dir data/raw/pdfs] [--persist-dir data/vectorstore/chroma_db]

The index is rebuilt from scratch on every run, so re-running never creates
duplicate chunks. Defaults come from config.py.
"""
import argparse
import shutil
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import Config
from pdf_loader import load_and_split_pdfs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", default=str(Config.PDF_DIR))
    parser.add_argument("--persist-dir", default=str(Config.VECTORSTORE_DIR))
    parser.add_argument("--chunk-size", type=int, default=Config.CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=Config.CHUNK_OVERLAP)
    args = parser.parse_args()

    pdfs = sorted(Path(args.pdf_dir).glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {args.pdf_dir}. Run `python gen_pdfs.py` for a sample.")

    chunks = load_and_split_pdfs(pdfs, args.chunk_size, args.chunk_overlap)

    persist = Path(args.persist_dir)
    if persist.exists():
        shutil.rmtree(persist)

    embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
    Chroma.from_documents(chunks, embeddings, persist_directory=str(persist))

    print(f"Ingested {len(pdfs)} PDF(s) -> {len(chunks)} chunks. Index saved at {persist}")


if __name__ == "__main__":
    main()
