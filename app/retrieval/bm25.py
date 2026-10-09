from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, documents=None):
        self.documents = []
        self.bm25 = None
        self.refresh(documents or [])

    def refresh(self, documents):
        self.documents = list(documents)

        if not self.documents:
            self.bm25 = None
            return

        tokenized_documents = [
            document["text"].lower().split()
            for document in self.documents
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

    def retrieve(self, query: str, top_k: int = 20):
        if not self.documents or self.bm25 is None or top_k <= 0:
            return []

        tokenized_query = query.lower().split()

        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        return [
            self.documents[i]
            for i in ranked_indices[:top_k]
            if scores[i] > 0
        ]