import pytest

from retrieval import BM25, Chunk, HybridRetriever, reciprocal_rank_fusion, tokenize


class FakeStore:
    """Minimal stand-in for a Chroma store. Dense search returns a fixed order."""

    def __init__(self, chunks, dense_order):
        self.chunks = chunks
        self.dense_order = dense_order

    def similarity_search(self, query, k=4):
        return [self.chunks[i] for i in self.dense_order][:k]

    def get(self, include=None):
        return {
            "documents": [c.page_content for c in self.chunks],
            "metadatas": [c.metadata for c in self.chunks],
        }


class FakeReranker:
    def predict(self, pairs):
        return [1.0 if "ZX-9981" in text else 0.0 for _, text in pairs]


@pytest.fixture
def store():
    chunks = [
        Chunk("general overview of the supply chain", {"source": "a.pdf", "page": 0}),
        Chunk("quarterly logistics summary and costs", {"source": "a.pdf", "page": 1}),
        Chunk("part serial ZX-9981 shipped on time", {"source": "a.pdf", "page": 2}),
        Chunk("supplier certificates and compliance", {"source": "a.pdf", "page": 3}),
    ]
    # Dense search is "blind" to the exact serial number: it ranks chunk 2 last
    return FakeStore(chunks, dense_order=[0, 1, 3, 2])


def test_tokenize_splits_on_punctuation_and_lowercases():
    assert tokenize("ZX-9981, Shipped!") == ["zx", "9981", "shipped"]


def test_bm25_ranks_exact_term_first_and_drops_zero_scores():
    bm25 = BM25(["alpha beta", "serial ZX-9981 part", "gamma delta"])
    assert bm25.top_n("ZX-9981", 3) == [1]


def test_bm25_empty_query_returns_nothing():
    assert BM25(["alpha beta"]).top_n("!!!", 3) == []


def test_bm25_handles_empty_corpus():
    assert BM25([]).top_n("anything", 3) == []


def test_rrf_rewards_items_ranked_well_in_both_lists():
    assert reciprocal_rank_fusion([["a", "b"], ["b", "c"]]) == ["b", "a", "c"]


def test_dense_mode_misses_exact_term(store):
    retriever = HybridRetriever(store, k=2, mode="dense")
    contents = [d.page_content for d in retriever.retrieve("ZX-9981")]
    assert not any("ZX-9981" in c for c in contents)


def test_hybrid_mode_recovers_exact_term(store):
    retriever = HybridRetriever(store, k=3, mode="hybrid")
    contents = [d.page_content for d in retriever.retrieve("ZX-9981")]
    assert any("ZX-9981" in c for c in contents)


def test_rerank_mode_puts_best_chunk_first(store):
    retriever = HybridRetriever(store, k=3, mode="rerank", reranker=FakeReranker())
    top = retriever.retrieve("ZX-9981")[0]
    assert "ZX-9981" in top.page_content


def test_retrieve_respects_k_override(store):
    retriever = HybridRetriever(store, k=3, mode="hybrid")
    assert len(retriever.retrieve("supply chain", k=1)) == 1


def test_invalid_mode_raises(store):
    with pytest.raises(ValueError):
        HybridRetriever(store, mode="magic")
