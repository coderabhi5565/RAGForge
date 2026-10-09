
import sys
import gc
import json
from pathlib import Path

# Add the RAGForge project root to Python's import path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.retrieval.service import RAGService


DATASET_PATH = ROOT / "evaluation" / "dataset" / "rag_eval.json"


def evaluate(use_query_rewriting: bool):
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    service = RAGService(
        use_query_rewriting=use_query_rewriting
    )

    hits_at_1 = []
    hits_at_3 = []
    hits_at_5 = []
    reciprocal_ranks = []

    try:
        for index, item in enumerate(dataset, start=1):
            question = item["question"]
            relevant_ids = set(item["relevant_chunk_ids"])

            documents = service.retrieve(
                question=question,
                top_k=5,
            )

            retrieved_ids = [
                doc.payload["metadata"]["chunk_id"]
                for doc in documents
            ]

            hits_at_1.append(
                int(any(cid in relevant_ids for cid in retrieved_ids[:1]))
            )
            hits_at_3.append(
                int(any(cid in relevant_ids for cid in retrieved_ids[:3]))
            )
            hits_at_5.append(
                int(any(cid in relevant_ids for cid in retrieved_ids[:5]))
            )

            reciprocal_rank = 0.0

            for rank, chunk_id in enumerate(retrieved_ids, start=1):
                if chunk_id in relevant_ids:
                    reciprocal_rank = 1.0 / rank
                    break

            reciprocal_ranks.append(reciprocal_rank)

            mode = "ON" if use_query_rewriting else "OFF"
            print(f"[{mode}] Query {index}/{len(dataset)} complete")

    finally:
        del service
        gc.collect()

    total = len(dataset)

    if total == 0:
        raise ValueError("Evaluation dataset is empty.")

    return {
        "queries": total,
        "Hit@1": sum(hits_at_1) / total,
        "Hit@3": sum(hits_at_3) / total,
        "Hit@5": sum(hits_at_5) / total,
        "MRR": sum(reciprocal_ranks) / total,
    }


def main():
    print("\n========== QUERY REWRITING: ON ==========")
    with_rewriting = evaluate(use_query_rewriting=True)

    print("\n========== QUERY REWRITING: OFF ==========")
    without_rewriting = evaluate(use_query_rewriting=False)

    print("\n========== COMPARISON RESULTS ==========")
    print(f"{'Metric':<12}{'ON':>12}{'OFF':>12}{'Delta':>12}")

    for metric in ("Hit@1", "Hit@3", "Hit@5", "MRR"):
        on_value = with_rewriting[metric]
        off_value = without_rewriting[metric]

        print(
            f"{metric:<12}"
            f"{on_value:>12.4f}"
            f"{off_value:>12.4f}"
            f"{on_value - off_value:>+12.4f}"
        )

    print(f"\nQueries per configuration: {with_rewriting['queries']}")
    print("Delta = ON - OFF")


if __name__ == "__main__":
    main()
