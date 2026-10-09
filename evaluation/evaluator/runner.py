
import json

from app.retrieval.service import RAGService
from evaluation.evaluator.retrieval import RetrievalEvaluator


class EvaluationRunner:

    def __init__(self):
        self.rag_service = RAGService()
        self.evaluator = RetrievalEvaluator(self.rag_service)

    def load_dataset(self, path: str):
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    def run(self, dataset_path: str):
        dataset = self.load_dataset(dataset_path)
        results = []

        for item in dataset:
            question = item["question"]
            relevant_chunk_ids = set(item["relevant_chunk_ids"])

            retrieved_docs = self.rag_service.retrieve(
                question=question,
                top_k=5,
            )

            retrieved_ids = [
                doc.payload["metadata"]["chunk_id"]
                for doc in retrieved_docs
            ]

            def hit_at_k(k):
                return int(
                    any(
                        chunk_id in relevant_chunk_ids
                        for chunk_id in retrieved_ids[:k]
                    )
                )

            reciprocal_rank = 0.0

            for rank, chunk_id in enumerate(retrieved_ids, start=1):
                if chunk_id in relevant_chunk_ids:
                    reciprocal_rank = 1.0 / rank
                    break

            results.append({
                "id": item["id"],
                "question": question,
                "hit_at_1": {"hit": hit_at_k(1)},
                "hit_at_3": {"hit": hit_at_k(3)},
                "hit_at_5": {"hit": hit_at_k(5)},
                "mrr": {"reciprocal_rank": reciprocal_rank},
            })

        return results

    def calculate_metrics(self, results):
        total = len(results)

        if total == 0:
            return {
                "Hit@1": 0.0,
                "Hit@3": 0.0,
                "Hit@5": 0.0,
                "MRR": 0.0,
            }

        return {
            "Hit@1": sum(r["hit_at_1"]["hit"] for r in results) / total,
            "Hit@3": sum(r["hit_at_3"]["hit"] for r in results) / total,
            "Hit@5": sum(r["hit_at_5"]["hit"] for r in results) / total,
            "MRR": sum(
                r["mrr"]["reciprocal_rank"] for r in results
            ) / total,
        }

    def run_diagnostics(self, dataset_path: str, k: int = 3):
        dataset = self.load_dataset(dataset_path)
        diagnostics = []

        for item in dataset:
            retrieved_docs = self.rag_service.retrieve(
                question=item["question"],
                top_k=k,
            )

            retrieved = [
                {
                    "rank": rank,
                    "chunk_id": doc.payload["metadata"]["chunk_id"],
                }
                for rank, doc in enumerate(retrieved_docs, start=1)
            ]

            relevant_chunk_ids = item["relevant_chunk_ids"]

            diagnostics.append({
                "id": item["id"],
                "question": item["question"],
                "relevant_chunk_ids": relevant_chunk_ids,
                "hit_at_1": int(
                    bool(retrieved)
                    and retrieved[0]["chunk_id"] in relevant_chunk_ids
                ),
                "retrieved": retrieved,
            })

        return diagnostics
