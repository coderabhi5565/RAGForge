
import json
from pathlib import Path

from app.retrieval.service import RAGService


class EvaluationRunner:
    K_VALUES = (1, 3, 5)

    def __init__(self):
        self.rag_service = RAGService()

    def load_dataset(self, path: str):
        with open(path, "r", encoding="utf-8") as file:
            dataset = json.load(file)

        if not isinstance(dataset, list):
            raise ValueError("Evaluation dataset must be a JSON list.")

        for index, item in enumerate(dataset):
            required = {"id", "question", "relevant_chunk_ids"}
            missing = required - item.keys()

            if missing:
                raise ValueError(
                    f"Dataset item at index {index} is missing: "
                    f"{sorted(missing)}"
                )

            if not isinstance(item["relevant_chunk_ids"], list):
                raise ValueError(
                    f"relevant_chunk_ids must be a list for item {item['id']}."
                )

        return dataset

    @staticmethod
    def _metrics_at_k(retrieved_ids, relevant_ids, k):
        """
        Calculate retrieval metrics for one query at cutoff k.

        Hit@K:
            1 if at least one relevant chunk is retrieved.

        Recall@K:
            Relevant retrieved chunks / all known relevant chunks.

        Precision@K:
            Relevant retrieved chunks / K.

        Duplicate chunk IDs are counted only once.
        """
        retrieved_at_k = retrieved_ids[:k]
        relevant_retrieved = len(
            set(retrieved_at_k) & relevant_ids
        )

        hit = int(relevant_retrieved > 0)

        recall = (
            relevant_retrieved / len(relevant_ids)
            if relevant_ids
            else 0.0
        )

        precision = relevant_retrieved / k

        return {
            "hit": hit,
            "recall": recall,
            "precision": precision,
        }

    @staticmethod
    def _reciprocal_rank(retrieved_ids, relevant_ids):
        for rank, chunk_id in enumerate(retrieved_ids, start=1):
            if chunk_id in relevant_ids:
                return 1.0 / rank

        return 0.0

    def run(self, dataset_path: str):
        dataset = self.load_dataset(dataset_path)
        results = []

        for item in dataset:
            question = item["question"]
            relevant_ids = set(item["relevant_chunk_ids"])

            retrieved_docs = self.rag_service.retrieve(
                question=question,
                top_k=max(self.K_VALUES),
            )

            retrieved_ids = [
                doc.payload["metadata"]["chunk_id"]
                for doc in retrieved_docs
            ]

            # Defensive check: retrieval should not return duplicate chunks.
            if len(retrieved_ids) != len(set(retrieved_ids)):
                raise ValueError(
                    f"Duplicate chunk IDs returned for query {item['id']}."
                )

            result = {
                "id": item["id"],
                "question": question,
                "relevant_chunk_ids": sorted(relevant_ids),
                "retrieved_chunk_ids": retrieved_ids,
            }

            for k in self.K_VALUES:
                metrics = self._metrics_at_k(
                    retrieved_ids=retrieved_ids,
                    relevant_ids=relevant_ids,
                    k=k,
                )

                result[f"hit_at_{k}"] = {"hit": metrics["hit"]}
                result[f"recall_at_{k}"] = {"recall": metrics["recall"]}
                result[f"precision_at_{k}"] = {
                    "precision": metrics["precision"]
                }

            result["mrr"] = {
                "reciprocal_rank": self._reciprocal_rank(
                    retrieved_ids, relevant_ids
                )
            }

            results.append(result)

        return results

    def calculate_metrics(self, results):
        total = len(results)

        metric_names = [
            *(f"Hit@{k}" for k in self.K_VALUES),
            *(f"Recall@{k}" for k in self.K_VALUES),
            *(f"Precision@{k}" for k in self.K_VALUES),
            "MRR",
        ]

        if total == 0:
            return {name: 0.0 for name in metric_names}

        metrics = {}

        for k in self.K_VALUES:
            metrics[f"Hit@{k}"] = sum(
                result[f"hit_at_{k}"]["hit"]
                for result in results
            ) / total

            metrics[f"Recall@{k}"] = sum(
                result[f"recall_at_{k}"]["recall"]
                for result in results
            ) / total

            metrics[f"Precision@{k}"] = sum(
                result[f"precision_at_{k}"]["precision"]
                for result in results
            ) / total

        metrics["MRR"] = sum(
            result["mrr"]["reciprocal_rank"]
            for result in results
        ) / total

        return metrics

    def run_diagnostics(self, dataset_path: str, k: int = 3):
        if k < 1:
            raise ValueError("k must be at least 1.")

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

            relevant_ids = set(item["relevant_chunk_ids"])
            retrieved_ids = [doc["chunk_id"] for doc in retrieved]

            metrics = self._metrics_at_k(
                retrieved_ids=retrieved_ids,
                relevant_ids=relevant_ids,
                k=k,
            )

            diagnostics.append({
                "id": item["id"],
                "question": item["question"],
                "relevant_chunk_ids": sorted(relevant_ids),
                "retrieved": retrieved,
                f"hit_at_{k}": metrics["hit"],
                f"recall_at_{k}": metrics["recall"],
                f"precision_at_{k}": metrics["precision"],
            })

        return diagnostics
