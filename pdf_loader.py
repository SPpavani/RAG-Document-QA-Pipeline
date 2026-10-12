"""Load PDFs and split them into overlapping chunks.

Used by ingest.py. You can also run it on its own to check a folder of PDFs:
    python pdf_loader.py [folder]
"""
import sys
from pathlib import Path
from typing import Iterable, List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Config


def load_and_split_pdfs(
    pdf_paths: Iterable,
    chunk_size: int = Config.CHUNK_SIZE,
    chunk_overlap: int = Config.CHUNK_OVERLAP,
) -> List:
    """Load every PDF page by page, then split into chunks (page metadata is kept)."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: List = []
    for path in pdf_paths:
        pages = PyPDFLoader(str(path)).load()
        chunks.extend(splitter.split_documents(pages))
    return chunks


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Config.PDF_DIR
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {folder}. Run `python gen_pdfs.py` for a sample.")
    print(f"Loaded {len(load_and_split_pdfs(pdfs))} chunks from {len(pdfs)} PDF(s)")
