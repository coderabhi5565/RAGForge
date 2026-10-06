from app.ingestion.loader import load_document
from app.ingestion.chunker import DocumentChunker
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector_store import VectorStore


class IngestionService:

    def __init__(self):
        self.chunker = DocumentChunker()
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()

    def ingest(self, file_path: str):
        documents = load_document(file_path)

        chunks = self.chunker.split_documents(documents)

        texts = [
            chunk.page_content
            for chunk in chunks
        ]

        embeddings = self.embedding_service.embed_documents(texts)

        self.vector_store.create_collection()

        self.vector_store.add_documents(
            chunks,
            embeddings,
        )

        return {
            "documents": len(documents),
            "chunks": len(chunks),
        }