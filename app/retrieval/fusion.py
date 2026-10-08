class RRFFusion:

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(
        self,
        rankings,
        top_k: int = 20,
    ):
        scores = {}
        documents = {}

        for ranking in rankings:
            for rank, document in enumerate(
                ranking,
                start=1,
            ):
                chunk_id = document["metadata"]["chunk_id"]

                if chunk_id not in scores:
                    scores[chunk_id] = 0.0
                    documents[chunk_id] = document

                scores[chunk_id] += (
                    1 / (self.k + rank)
                )

        ranked_chunk_ids = sorted(
            scores,
            key=scores.get,
            reverse=True,
        )

        return [
            (
                documents[chunk_id],
                scores[chunk_id],
            )
            for chunk_id in ranked_chunk_ids[:top_k]
        ]