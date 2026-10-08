from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector_store import VectorStore
from app.generation.generator import Generator


class RAGService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.generator = Generator()

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
    ):
        query_vector = self.embedding_service.embed_query(
            question
        )

        results = self.vector_store.similarity_search(
            query_vector=query_vector,
            top_k=top_k,
        )

        return results

    def query(self,question: str,top_k: int = 5,):
        results = self.retrieve(
            question=question,
            top_k=top_k,
        )

        context = [
            result.payload["text"]
            for result in results
        ]

        answer = self.generator.generate(
            query=question,
            context=context,
        )

        return {
            "answer": answer,
            "sources": [
                result.payload["metadata"]
                for result in results
            ],
        }