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
            relevant_chunk_ids = item["relevant_chunk_ids"]

            question_result = {
                "id": item["id"],
                "question": question,
                "hit_at_1": self.evaluator.evaluate_hit_at_k(
                    question,
                    relevant_chunk_ids,
                    1,
                ),
                "hit_at_3": self.evaluator.evaluate_hit_at_k(
                    question,
                    relevant_chunk_ids,
                    3,
                ),
                "hit_at_5": self.evaluator.evaluate_hit_at_k(
                    question,
                    relevant_chunk_ids,
                    5,
                ),
                "mrr": self.evaluator.evaluate_mrr(
                    question,
                    relevant_chunk_ids,
                    5,
                ),
            }

            results.append(question_result)

        return results

    def calculate_metrics(self, results):
        total = len(results)

        hit_at_1 = sum(
            result["hit_at_1"]["hit"]
            for result in results
        ) / total

        hit_at_3 = sum(
            result["hit_at_3"]["hit"]
            for result in results
        ) / total

        hit_at_5 = sum(
            result["hit_at_5"]["hit"]
            for result in results
        ) / total

        mrr = sum(
            result["mrr"]["reciprocal_rank"]
            for result in results
        ) / total

        return {
            "Hit@1": hit_at_1,
            "Hit@3": hit_at_3,
            "Hit@5": hit_at_5,
            "MRR": mrr,
        }

    def run_diagnostics(self, dataset_path: str, k: int = 3):
        dataset = self.load_dataset(dataset_path)

        diagnostics = []

        for item in dataset:
            results = self.rag_service.retrieve(
                question=item["question"],
                top_k=k,
            )

            retrieved = []

            for rank, result in enumerate(results, start=1):
                retrieved.append({
                    "rank": rank,
                    "chunk_id": result.payload["metadata"]["chunk_id"],
                    "score": result.score,
                })

            relevant_chunk_ids = item["relevant_chunk_ids"]

            hit_at_1 = (
                retrieved[0]["chunk_id"] in relevant_chunk_ids
            )

            diagnostics.append({
                "id": item["id"],
                "question": item["question"],
                "relevant_chunk_ids": relevant_chunk_ids,
                "hit_at_1": int(hit_at_1),
                "retrieved": retrieved,
            })

        return diagnostics