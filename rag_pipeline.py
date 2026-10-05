"""
Core RAG pipeline module.
Combines document retrieval with LLM-based generation.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """Use only the following pieces of context to answer the question at the end.
If the answer is not in the context, say that you don't know. Do not make up an answer.

{context}

Question: {question}
Answer:"""


def _format_source(doc) -> Dict[str, Any]:
    """Turn a retrieved document into a citation (file name, 1-based page, snippet)."""
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
    ):
        """
        Args:
            vectorstore_path: Path to the Chroma database
            embedding_model: HuggingFace embedding model name
            llm_model: OpenAI chat model name
            search_k: Number of chunks to retrieve
            openai_api_key: OpenAI API key (falls back to OPENAI_API_KEY env var)
            temperature: 0 keeps answers deterministic, which suits factual Q&A
        """
        self.vectorstore_path = vectorstore_path
        self.embedding_model_name = embedding_model
        self.llm_model = llm_model
        self.search_k = search_k
        self.temperature = temperature

        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None

        self._initialize(openai_api_key)

    def _initialize(self, openai_api_key: Optional[str] = None):
        """Initialize embeddings, vector store and the QA chain."""
        try:
            logger.info(f"Loading embeddings: {self.embedding_model_name}")
            self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

            logger.info(f"Loading vectorstore from: {self.vectorstore_path}")
            self.vectorstore = Chroma(
                persist_directory=self.vectorstore_path,
                embedding_function=self.embeddings,
            )
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.search_k}
            )

            logger.info(f"Initializing LLM: {self.llm_model}")
            llm = ChatOpenAI(
                model_name=self.llm_model,
                api_key=openai_api_key,
                temperature=self.temperature,
            )

            prompt = PromptTemplate(
                template=PROMPT_TEMPLATE,
                input_variables=["context", "question"],
            )
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.retriever,
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True,
            )
            logger.info("RAG pipeline initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing RAG pipeline: {e}")
            raise

    def query(self, question: str) -> Dict[str, Any]:
        """Run a RAG query.

        Returns:
            {"answer": str, "sources": [{content, source, page}], "source_count": int}
        """
        if not self.qa_chain:
            raise RuntimeError("RAG pipeline not initialized")
        if not question or not question.strip():
            raise ValueError("Question must not be empty")

        try:
            logger.info(f"Processing query: {question}")
            result = self.qa_chain.invoke({"query": question})
            sources = [_format_source(doc) for doc in result["source_documents"]]
            return {
                "answer": result["result"],
                "sources": sources,
                "source_count": len(sources),
            }
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise

    def retrieve_documents(self, query: str, k: Optional[int] = None) -> List[Dict]:
        """Retrieve relevant chunks without calling the LLM."""
        if not self.retriever:
            raise RuntimeError("Retriever not initialized")

        k = k or self.search_k
        try:
            docs = self.retriever.invoke(query)
            return [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in docs[:k]
            ]
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            raise


def setup_logging():
    """Configure logging for the RAG pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
