import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from app.retrieval.service import RAGService


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "evaluation" / "dataset" / "rag_eval.json"
OUTPUT_PATH = ROOT / "evaluation" / "results" / "retrieval_ablation.json"

K_VALUES = (1, 3, 5)
CANDIDATE_K = 10
FINAL_K = 5


class RerankDocument:
    """Adapt a retrieval dictionary to the reranker's expected interface."""

    def __init__(self, document):
        self.payload = {
            "text": document["text"],
            "metadata": document["metadata"],
        }


def load_dataset():
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list) or not dataset:
        raise ValueError("Evaluation dataset must be a non-empty JSON list.")

    for item in dataset:
        if not item.get("id") or not item.get("question"):
            raise ValueError("Every dataset item needs an id and question.")

        if not isinstance(item.get("relevant_chunk_ids"), list):
            raise ValueError(
                f"Query {item.get('id')} must have relevant_chunk_ids as a list."
            )

    return dataset


def dense_documents(rag, query):
    vector = rag.embedding_service.embed_query(query)
    points = rag.vector_store.similarity_search(
        query_vector=vector,
        top_k=CANDIDATE_K,
    )

    return [
        {
            "text": point.payload["text"],
            "metadata": point.payload["metadata"],
        }
        for point in points
        if point.payload
        and point.payload.get("text")
        and point.payload.get("metadata")
        and point.payload["metadata"].get("chunk_id")
    ]


def bm25_documents(rag, query):
    return rag.bm25.retrieve(
        query=query,
        top_k=CANDIDATE_K,
    )


def hybrid_documents(rag, query):
    dense = dense_documents(rag, query)
    sparse = bm25_documents(rag, query)

    return rag.fusion.fuse(
        rankings=[dense, sparse],
        top_k=CANDIDATE_K,
    )


def retrieve_for_experiment(rag, experiment, question):
    if experiment == "A_dense_only":
        return dense_documents(rag, question)[:FINAL_K]

    if experiment == "B_bm25_only":
        return bm25_documents(rag, question)[:FINAL_K]

    if experiment == "C_hybrid_rrf":
        return hybrid_documents(rag, question)[:FINAL_K]

    if experiment == "D_hybrid_reranking":
        candidates = hybrid_documents(rag, question)

        if not candidates:
            return []

        return rag.reranker.rerank(
            query=question,
            documents=[
                RerankDocument(document)
                for document in candidates
            ],
            top_k=FINAL_K,
        )

    if experiment == "E_full_adaptive":
        # Uses the existing production retrieval pipeline unchanged.
        return rag.retrieve(
            question=question,
            top_k=FINAL_K,
        )

    raise ValueError(f"Unknown experiment: {experiment}")


def chunk_id(document):
    # Full adaptive and reranked results expose .payload;
    # dense/BM25/RRF results are dictionaries.
    payload = (
        document.payload
        if hasattr(document, "payload")
        else document
    )

    return payload.get("metadata", {}).get("chunk_id")


def percentile(values, percent):
    if not values:
        return 0.0

    ordered = sorted(values)
    index = max(0, math.ceil(percent / 100 * len(ordered)) - 1)
    return round(ordered[index], 2)


def calculate_metrics(query_results):
    metrics = {}

    for k in K_VALUES:
        hits = []
        recalls = []
        precisions = []

        for result in query_results:
            relevant = set(result["relevant_chunk_ids"])
            retrieved = result["retrieved_chunk_ids"][:k]
            retrieved_set = set(retrieved)
            relevant_found = len(retrieved_set & relevant)

            hits.append(
                1.0 if retrieved_set & relevant else 0.0
            )
            recalls.append(
                relevant_found / len(relevant) if relevant else 0.0
            )
            # Standard P@K: denominator remains k, even if fewer
            # than k documents are returned.
            precisions.append(relevant_found / k)

        metrics[f"hit_at_{k}"] = round(sum(hits) / len(hits), 4)
        metrics[f"recall_at_{k}"] = round(
            sum(recalls) / len(recalls), 4
        )
        metrics[f"precision_at_{k}"] = round(
            sum(precisions) / len(precisions), 4
        )

    reciprocal_ranks = []

    for result in query_results:
        relevant = set(result["relevant_chunk_ids"])
        rank = next(
            (
                index
                for index, cid in enumerate(
                    result["retrieved_chunk_ids"], start=1
                )
                if cid in relevant
            ),
            None,
        )
        reciprocal_ranks.append(1 / rank if rank else 0.0)

    latencies = [result["latency_ms"] for result in query_results]

    metrics["mrr"] = round(
        sum(reciprocal_ranks) / len(reciprocal_ranks), 4
    )
    metrics["latency_ms_mean"] = round(
        sum(latencies) / len(latencies), 2
    )
    metrics["latency_ms_p50"] = percentile(latencies, 50)
    metrics["latency_ms_p95"] = percentile(latencies, 95)

    return metrics


def main():
    dataset = load_dataset()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # One shared instance: same models, Qdrant collection and BM25 index.
    rag = RAGService(use_query_rewriting=False)

    experiments = [
        "A_dense_only",
        "B_bm25_only",
        "C_hybrid_rrf",
        "D_hybrid_reranking",
        "E_full_adaptive",
    ]

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(DATASET_PATH.relative_to(ROOT)),
        "query_count": len(dataset),
        "candidate_k": CANDIDATE_K,
        "final_k": FINAL_K,
        "query_rewriting": False,
        "experiments": {},
    }

    for experiment in experiments:
        print(f"\nRunning: {experiment}")
        query_results = []

        for index, item in enumerate(dataset, start=1):
            question = item["question"]
            started = time.perf_counter()

            documents = retrieve_for_experiment(
                rag, experiment, question
            )

            latency_ms = (time.perf_counter() - started) * 1000

            retrieved_ids = []
            for document in documents[:FINAL_K]:
                cid = chunk_id(document)
                if cid is not None:
                    retrieved_ids.append(cid)

            query_results.append({
                "id": item["id"],
                "question": question,
                "relevant_chunk_ids": item["relevant_chunk_ids"],
                "retrieved_chunk_ids": retrieved_ids,
                "latency_ms": round(latency_ms, 2),
            })

            print(
                f"  [{index}/{len(dataset)}] "
                f"retrieved={len(retrieved_ids)}, "
                f"latency={latency_ms:.1f} ms"
            )

        report["experiments"][experiment] = {
            "metrics": calculate_metrics(query_results),
            "queries": query_results,
        }

        print(json.dumps(
            report["experiments"][experiment]["metrics"],
            indent=2,
        ))

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print(f"\nReport saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()