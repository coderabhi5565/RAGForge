from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self, documents):
        self.documents = documents

        tokenized_documents = [
            document["text"].lower().split()
            for document in documents
        ]

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
    ):
        tokenized_query = query.lower().split()

        scores = self.bm25.get_scores(
            tokenized_query
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        return [
            self.documents[i]
            for i in ranked_indices[:top_k]
        ]