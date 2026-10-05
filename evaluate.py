"""Retrieval evaluation: hit-rate@k, MRR and latency. No LLM or API key needed.

A retrieved chunk counts as relevant when it contains ALL expected keywords for
the question (case-insensitive, whitespace-normalised). This is a cheap,
deterministic proxy for retrieval quality.

Usage:
    python evaluate.py [--questions eval/questions.json] [--k 3]
"""
import argparse
import json
import time
from pathlib import Path
from typing import List, Optional


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def first_hit_rank(chunks: List[str], keywords: List[str]) -> Optional[int]:
    """1-based rank of the first chunk containing all keywords, else None."""
    kws = [normalize(k) for k in keywords]
    for rank, chunk in enumerate(chunks, start=1):
        text = normalize(chunk)
        if all(k in text for k in kws):
            return rank
    return None


def hit_rate(ranks: List[Optional[int]], k: int) -> float:
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mrr(ranks: List[Optional[int]]) -> float:
    if not ranks:
        return 0.0
    return sum(1.0 / r for r in ranks if r) / len(ranks)


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(p / 100 * (len(ordered) - 1))))
    return ordered[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default="eval/questions.json")
    parser.add_argument("--vectorstore", default="data/vectorstore/chroma_db")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    # Heavy imports only when actually running the evaluation
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=args.vectorstore, embedding_function=embeddings)

    ranks, latencies, rows = [], [], []
    for item in questions:
        start = time.perf_counter()
        docs = db.similarity_search(item["question"], k=args.k)
        latencies.append((time.perf_counter() - start) * 1000)
        rank = first_hit_rank([d.page_content for d in docs], item["expected_keywords"])
        ranks.append(rank)
        rows.append({"question": item["question"], "rank": rank})
        print(f"[{'HIT ' if rank else 'MISS'}] rank={rank}  {item['question']}")

    summary = {
        "questions": len(questions),
        f"hit_rate@{args.k}": round(hit_rate(ranks, args.k), 3),
        "mrr": round(mrr(ranks), 3),
        "latency_p50_ms": round(percentile(latencies, 50), 1),
        "latency_p95_ms": round(percentile(latencies, 95), 1),
    }
    print("\n" + json.dumps(summary, indent=2))

    out = Path("eval/results.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "details": rows}, indent=2), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
