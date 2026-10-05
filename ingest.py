"""Ingest every PDF in a folder into a persistent Chroma vector store.

Usage:
    python ingest.py [--pdf-dir data/raw/pdfs] [--persist-dir data/vectorstore/chroma_db]

The index is rebuilt from scratch on every run, so re-running never creates
duplicate chunks.
"""
import argparse
import shutil
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", default="data/raw/pdfs")
    parser.add_argument("--persist-dir", default="data/vectorstore/chroma_db")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    args = parser.parse_args()

    pdfs = sorted(Path(args.pdf_dir).glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {args.pdf_dir}. Run `python gen_pdfs.py` for a sample.")

    docs = []
    for pdf in pdfs:
        docs.extend(PyPDFLoader(str(pdf)).load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )
    chunks = splitter.split_documents(docs)

    persist = Path(args.persist_dir)
    if persist.exists():
        shutil.rmtree(persist)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    Chroma.from_documents(chunks, embeddings, persist_directory=str(persist))

    print(f"Ingested {len(pdfs)} PDF(s) -> {len(chunks)} chunks. Index saved at {persist}")


if __name__ == "__main__":
    main()
