class RetrievalEvaluator:

    def __init__(self, rag_service):
        self.rag_service = rag_service

    def evaluate_hit_at_k(
        self,
        question: str,
        relevant_chunk_ids: list[str],
        k: int,
    ):

        results = self.rag_service.retrieve(
            question=question,
            top_k=k,
        )

        retrieved_chunk_ids = [
            result.payload["metadata"]["chunk_id"]
            for result in results
        ]

        hit = any(
            chunk_id in relevant_chunk_ids
            for chunk_id in retrieved_chunk_ids
        )

        return {
            "hit": int(hit),
            "retrieved_chunk_ids": retrieved_chunk_ids,
        }

    def evaluate_mrr(
        self,
        question: str,
        relevant_chunk_ids: list[str],
        k: int = 5,
    ):

        results = self.rag_service.retrieve(
            question=question,
            top_k=k,
        )

        retrieved_chunk_ids = [
            result.payload["metadata"]["chunk_id"]
            for result in results
        ]

        reciprocal_rank = 0.0

        for rank, chunk_id in enumerate(
            retrieved_chunk_ids,
            start=1,
        ):
            if chunk_id in relevant_chunk_ids:
                reciprocal_rank = 1 / rank
                break

        return {
            "reciprocal_rank": reciprocal_rank,
            "retrieved_chunk_ids": retrieved_chunk_ids,
        }    