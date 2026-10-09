from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector_store import VectorStore
from app.generation.generator import Generator
from app.retrieval.reranker import Reranker
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.fusion import RRFFusion
from app.retrieval.query_rewriter import QueryRewriter


class RAGService:
    def __init__(
        self,
        use_query_rewriting: bool = False,
        initial_candidate_k: int = 10,
        expanded_candidate_k: int = 30,
        agreement_threshold: int = 2,
    ):
        self.use_query_rewriting = use_query_rewriting
        self.initial_candidate_k = initial_candidate_k
        self.expanded_candidate_k = expanded_candidate_k
        self.agreement_threshold = agreement_threshold

        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.generator = Generator()
        self.reranker = Reranker()
        self.fusion = RRFFusion()
        self.query_rewriter = QueryRewriter()

        self.refresh_bm25()

    def refresh_bm25(self):
        documents = self.vector_store.get_all_documents()
        self.bm25 = BM25Retriever(documents)

    @staticmethod
    def _chunk_ids(documents):
        return {
            document["metadata"]["chunk_id"]
            for document in documents
        }

    def retrieve(self, question: str, top_k: int = 5):
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20.")

        retrieval_query = question.strip()

        if self.use_query_rewriting:
            rewritten_query = self.query_rewriter.rewrite(
                retrieval_query
            )
            if rewritten_query and rewritten_query.strip():
                retrieval_query = rewritten_query.strip()

        query_vector = self.embedding_service.embed_query(
            retrieval_query
        )

        initial_k = self.initial_candidate_k

        dense_points = self.vector_store.similarity_search(
            query_vector=query_vector,
            top_k=initial_k,
        )

        dense_results = [
            {
                "text": point.payload["text"],
                "metadata": point.payload["metadata"],
            }
            for point in dense_points
            if point.payload
            and point.payload.get("text")
            and point.payload.get("metadata") is not None
        ]

        bm25_results = self.bm25.retrieve(
            query=retrieval_query,
            top_k=initial_k,
        )

        dense_top_ids = self._chunk_ids(dense_results[:5])
        bm25_top_ids = self._chunk_ids(bm25_results[:5])
        overlap = len(dense_top_ids & bm25_top_ids)

        expanded = overlap < self.agreement_threshold

        if expanded:
            dense_points = self.vector_store.similarity_search(
                query_vector=query_vector,
                top_k=self.expanded_candidate_k,
            )

            dense_results = [
                {
                    "text": point.payload["text"],
                    "metadata": point.payload["metadata"],
                }
                for point in dense_points
                if point.payload
                and point.payload.get("text")
                and point.payload.get("metadata") is not None
            ]

            bm25_results = self.bm25.retrieve(
                query=retrieval_query,
                top_k=self.expanded_candidate_k,
            )

        fused_results = self.fusion.fuse(
            rankings=[dense_results, bm25_results],
            top_k=self.expanded_candidate_k if expanded else initial_k,
        )

        if not fused_results:
            return []

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

        return self.reranker.rerank(
            query=question,
            documents=rerank_documents,
            top_k=top_k,
        )

    def query(self, question: str, top_k: int = 5):
        results = self.retrieve(
            question=question,
            top_k=top_k,
        )

        if not results:
            return {
                "answer": (
                    "I don't have enough information in the "
                    "provided documents."
                ),
                "sources": [],
            }

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
