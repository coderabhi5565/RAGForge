from sentence_transformers import SentenceTransformer


class EmbeddingService:
    def __init__(self,model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model)

    def embed_documents(
        self,
        documents: list[str],
    ) -> list[list[float]]:

        embeddings = self.model.encode(
            documents,
            convert_to_numpy=True,
        )

        return embeddings.tolist()

    def embed_query(
        self,
        query: str,
    ) -> list[float]:

        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
        )

        return embedding.tolist()