"""Retrieval strategies: dense, hybrid (BM25 + vectors with RRF) and hybrid + rerank.

Why hybrid? Dense embeddings capture meaning but can blur exact terms such as
part numbers, names and acronyms. BM25 matches exact terms but misses
paraphrases. Reciprocal Rank Fusion (RRF) combines both rankings without having
to normalise their different score scales. A cross-encoder can then re-score the
shortlist by reading the query and each chunk together.

This module has no heavy imports at load time: BM25 and RRF are plain Python and
the cross-encoder is imported lazily, so the logic is unit-testable in CI.
"""
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

MODES = ("dense", "hybrid", "rerank")
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class Chunk:
    """Minimal stand-in for a LangChain Document (same attribute names)."""

    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BM25:
    """Okapi BM25 implemented from scratch (no extra dependency)."""

    def __init__(self, corpus: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(text) for text in corpus]
        self.n = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / self.n if self.n else 0.0
        self.freqs = [Counter(d) for d in self.docs]
        doc_freq: Counter = Counter()
        for d in self.docs:
            doc_freq.update(set(d))
        # "+1" inside the log keeps IDF non-negative
        self.idf = {
            term: math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }

    def scores(self, query: str) -> List[float]:
        terms = tokenize(query)
        out = []
        for freq, doc in zip(self.freqs, self.docs):
            doc_len = len(doc)
            score = 0.0
            for term in terms:
                tf = freq.get(term, 0)
                if not tf:
                    continue
                norm = 1 - self.b + self.b * doc_len / self.avgdl if self.avgdl else 1.0
                score += self.idf[term] * tf * (self.k1 + 1) / (tf + self.k1 * norm)
            out.append(score)
        return out

    def top_n(self, query: str, n: int) -> List[int]:
        """Indices of the best-matching documents (zero-score documents excluded)."""
        scores = self.scores(query)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [i for i in order[:n] if scores[i] > 0]


def reciprocal_rank_fusion(rankings: List[List[str]], k: int = 60) -> List[str]:
    """Fuse several ranked lists of keys into one. Higher fused score = better."""
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda key: scores[key], reverse=True)


def doc_key(doc) -> str:
    meta = doc.metadata or {}
    return f"{meta.get('source')}|{meta.get('page')}|{doc.page_content}"


class HybridRetriever:
    """Retrieve chunks from a Chroma store using dense, hybrid or rerank mode."""

    def __init__(
        self,
        vectorstore,
        k: int = 3,
        fetch_k: int = 10,
        mode: str = "hybrid",
        reranker=None,
        reranker_model: str = DEFAULT_RERANKER,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got '{mode}'")
        self.vectorstore = vectorstore
        self.k = k
        self.fetch_k = fetch_k
        self.mode = mode
        self._reranker = reranker
        self._reranker_model = reranker_model
        self._bm25: Optional[BM25] = None
        self._chunks: List[Chunk] = []

    def _load_corpus(self) -> None:
        if self._bm25 is not None:
            return
        data = self.vectorstore.get(include=["documents", "metadatas"])
        self._chunks = [
            Chunk(page_content=text, metadata=meta or {})
            for text, meta in zip(data["documents"], data["metadatas"])
        ]
        self._bm25 = BM25([c.page_content for c in self._chunks])

    def _rerank(self, query: str, docs: list) -> list:
        if self._reranker is None:
            from sentence_transformers import CrossEncoder  # lazy: heavy import

            self._reranker = CrossEncoder(self._reranker_model)
        scores = self._reranker.predict([(query, d.page_content) for d in docs])
        ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
        return [doc for doc, _ in ranked]

    def retrieve(self, query: str, k: Optional[int] = None) -> list:
        k = k or self.k
        if self.mode == "dense":
            return list(self.vectorstore.similarity_search(query, k=k))

        dense = list(self.vectorstore.similarity_search(query, k=self.fetch_k))
        self._load_corpus()
        sparse = [self._chunks[i] for i in self._bm25.top_n(query, self.fetch_k)]

        by_key: Dict[str, Any] = {}
        rankings: List[List[str]] = []
        for ranked_docs in (dense, sparse):
            keys = []
            for d in ranked_docs:
                key = doc_key(d)
                by_key.setdefault(key, d)
                keys.append(key)
            rankings.append(keys)

        fused = [by_key[key] for key in reciprocal_rank_fusion(rankings)]
        if self.mode == "rerank" and fused:
            fused = self._rerank(query, fused[: self.fetch_k])
        return fused[:k]
