"""
Core RAG pipeline module.
Combines retrieval (dense / hybrid / rerank) with LLM-based generation.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from retrieval import HybridRetriever

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """Use only the following pieces of context to answer the question at the end.
If the answer is not in the context, say that you don't know. Do not make up an answer.

{context}

Question: {question}
Answer:"""

NO_CONTEXT_ANSWER = "I don't know. No relevant context was found in the documents."


def _format_source(doc) -> Dict[str, Any]:
    """Turn a retrieved chunk into a citation (file name, 1-based page, snippet)."""
    meta = doc.metadata or {}
    page = meta.get("page")
    return {
        "content": doc.page_content[:300],
        "source": os.path.basename(meta.get("source", "")) or None,
        "page": page + 1 if isinstance(page, int) else None,
    }


class RAGPipeline:
    """Complete RAG (Retrieval-Augmented Generation) pipeline."""

    def __init__(
        self,
        vectorstore_path: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "gpt-3.5-turbo",
        search_k: int = 3,
        openai_api_key: Optional[str] = None,
        temperature: float = 0.0,
        retrieval_mode: str = "hybrid",
    ):
        """
        Args:
            vectorstore_path: Path to the Chroma database
            embedding_model: HuggingFace embedding model name
            llm_model: OpenAI chat model name
            search_k: Number of chunks passed to the LLM
            openai_api_key: OpenAI API key (falls back to OPENAI_API_KEY env var)
            temperature: 0 keeps answers deterministic, which suits factual Q&A
            retrieval_mode: "dense", "hybrid" (BM25 + vectors) or "rerank"
                            (hybrid + cross-encoder)
        """
        self.vectorstore_path = vectorstore_path
        self.embedding_model_name = embedding_model
        self.llm_model = llm_model
        self.search_k = search_k
        self.temperature = temperature
        self.retrieval_mode = retrieval_mode

        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self.llm = None

        self._initialize(openai_api_key)

    def _initialize(self, openai_api_key: Optional[str] = None):
        """Initialize embeddings, vector store, retriever and LLM."""
        try:
            logger.info(f"Loading embeddings: {self.embedding_model_name}")
            self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

            logger.info(f"Loading vectorstore from: {self.vectorstore_path}")
            self.vectorstore = Chroma(
                persist_directory=self.vectorstore_path,
                embedding_function=self.embeddings,
            )
            self.retriever = HybridRetriever(
                self.vectorstore, k=self.search_k, mode=self.retrieval_mode
            )

            logger.info(f"Initializing LLM: {self.llm_model}")
            self.llm = ChatOpenAI(
                model_name=self.llm_model,
                api_key=openai_api_key,
                temperature=self.temperature,
            )
            logger.info(f"RAG pipeline ready (retrieval mode: {self.retrieval_mode})")

        except Exception as e:
            logger.error(f"Error initializing RAG pipeline: {e}")
            raise

    def query(self, question: str) -> Dict[str, Any]:
        """Run a RAG query.

        Returns:
            {"answer": str, "sources": [{content, source, page}], "source_count": int}
        """
        if not self.llm or not self.retriever:
            raise RuntimeError("RAG pipeline not initialized")
        if not question or not question.strip():
            raise ValueError("Question must not be empty")

        try:
            logger.info(f"Processing query: {question}")
            docs = self.retriever.retrieve(question)
            if not docs:
                return {"answer": NO_CONTEXT_ANSWER, "sources": [], "source_count": 0}

            context = "\n\n".join(d.page_content for d in docs)
            prompt = PROMPT_TEMPLATE.format(context=context, question=question)
            answer = self.llm.invoke(prompt).content

            sources = [_format_source(d) for d in docs]
            return {"answer": answer, "sources": sources, "source_count": len(sources)}
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise

    def retrieve_documents(self, query: str, k: Optional[int] = None) -> List[Dict]:
        """Retrieve relevant chunks without calling the LLM."""
        if not self.retriever:
            raise RuntimeError("Retriever not initialized")
        try:
            docs = self.retriever.retrieve(query, k=k)
            return [{"content": d.page_content, "metadata": d.metadata} for d in docs]
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            raise


def setup_logging():
    """Configure logging for the RAG pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
