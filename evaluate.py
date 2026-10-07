"""Retrieval evaluation: compares dense, hybrid and rerank modes. No LLM or API key needed.

Metrics: hit-rate@k, MRR and latency (p50 / p95).

A retrieved chunk counts as relevant when it contains ALL expected keywords for
the question (case-insensitive, whitespace-normalised). This is a cheap,
deterministic proxy for retrieval quality.

Usage:
    python evaluate.py                       # compare all three modes
    python evaluate.py --modes dense hybrid  # choose modes
    python evaluate.py --k 3 --questions eval/questions.json
"""
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional


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


def summarize(ranks: List[Optional[int]], latencies: List[float], k: int) -> Dict:
    return {
        f"hit_rate@{k}": round(hit_rate(ranks, k), 3),
        "mrr": round(mrr(ranks), 3),
        "latency_p50_ms": round(percentile(latencies, 50), 1),
        "latency_p95_ms": round(percentile(latencies, 95), 1),
    }


def format_table(results: Dict[str, Dict], k: int) -> str:
    """Plain-text comparison table, one row per retrieval mode."""
    hit_key = f"hit_rate@{k}"
    lines = [
        f"{'mode':<10} {hit_key:>12} {'mrr':>7} {'p50 ms':>9} {'p95 ms':>9}",
        "-" * 50,
    ]
    for mode, s in results.items():
        lines.append(
            f"{mode:<10} {s[hit_key]:>12.3f} {s['mrr']:>7.3f} "
            f"{s['latency_p50_ms']:>9.1f} {s['latency_p95_ms']:>9.1f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default="eval/questions.json")
    parser.add_argument("--vectorstore", default="data/vectorstore/chroma_db")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--modes", nargs="+", default=["dense", "hybrid", "rerank"])
    args = parser.parse_args()

    # Heavy imports only when actually running the evaluation
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    from retrieval import HybridRetriever

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=args.vectorstore, embedding_function=embeddings)

    summaries: Dict[str, Dict] = {}
    details: Dict[str, List[Dict]] = {}

    for mode in args.modes:
        retriever = HybridRetriever(db, k=args.k, mode=mode)
        ranks, latencies, rows = [], [], []
        try:
            retriever.retrieve(questions[0]["question"])  # warm-up (model loading)
            for item in questions:
                start = time.perf_counter()
                docs = retriever.retrieve(item["question"])
                latencies.append((time.perf_counter() - start) * 1000)
                rank = first_hit_rank([d.page_content for d in docs], item["expected_keywords"])
                ranks.append(rank)
                rows.append({"question": item["question"], "rank": rank})
        except Exception as exc:  # e.g. reranker model could not be downloaded
            print(f"Skipping mode '{mode}': {exc}")
            continue
        summaries[mode] = summarize(ranks, latencies, args.k)
        details[mode] = rows

    print(f"\nQuestions: {len(questions)}   k = {args.k}\n")
    print(format_table(summaries, args.k))

    misses = {
        mode: [r["question"] for r in rows if r["rank"] is None]
        for mode, rows in details.items()
    }
    for mode, qs in misses.items():
        if qs:
            print(f"\nMissed in '{mode}':")
            for q in qs:
                print(f"  - {q}")

    out = Path("eval/results.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        json.dumps({"k": args.k, "summary": summaries, "details": details}, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
