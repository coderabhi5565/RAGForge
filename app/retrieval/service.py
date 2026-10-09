
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector_store import VectorStore
from app.generation.generator import Generator
from app.retrieval.reranker import Reranker
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.fusion import RRFFusion
from app.retrieval.query_rewriter import QueryRewriter


class RAGService:

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.generator = Generator()
        self.reranker = Reranker()
        self.fusion = RRFFusion()
        self.query_rewriter = QueryRewriter()

        documents = self.vector_store.get_all_documents()

        self.bm25 = BM25Retriever(documents)

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
    ):
        # Step 1: Rewrite the original question
        rewritten_question = self.query_rewriter.rewrite(question)

        # Step 2: Dense retrieval uses rewritten query
        query_vector = self.embedding_service.embed_query(
            rewritten_question
        )

        dense_results = self.vector_store.similarity_search(
            query_vector=query_vector,
            top_k=20,
        )

        dense_results = [
            {
                "text": result.payload["text"],
                "metadata": result.payload["metadata"],
            }
            for result in dense_results
        ]

        # Step 3: BM25 also uses rewritten query
        bm25_results = self.bm25.retrieve(
            query=rewritten_question,
            top_k=20,
        )

        # Step 4: Fuse dense and BM25 rankings
        fused_results = self.fusion.fuse(
            rankings=[
                dense_results,
                bm25_results,
            ],
            top_k=20,
        )

        # Step 5: Convert dictionaries for the reranker
        class RerankDocument:

            def __init__(self, document):
                self.payload = {
                    "text": document["text"],
                    "metadata": document["metadata"],
                }

        rerank_documents = [
            RerankDocument(document)
            for document in fused_results
        ]

        # Step 6: Reranker uses the original question
        results = self.reranker.rerank(
            query=question,
            documents=rerank_documents,
            top_k=top_k,
        )

        return results

    def query(
        self,
        question: str,
        top_k: int = 5,
    ):
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
