import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)
from langchain_core.documents import Document


class VectorStore:

    def __init__(
        self,
        collection_name: str = "ragforge_documents",
        vector_size: int = 384,
    ):
        self.collection_name = collection_name

        self.client = QdrantClient(
            url="http://localhost:6333"
        )

        self.vector_size = vector_size

    def create_collection(self):

        collections = self.client.get_collections()

        collection_names = [
            collection.name
            for collection in collections.collections
        ]

        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ):

        points = []

        for document, embedding in zip(
            documents,
            embeddings,
        ):
            chunk_id = document.metadata["chunk_id"]

            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    chunk_id,
                )
            )

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "text": document.page_content,
                        "metadata": document.metadata,
                    },
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def similarity_search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        source_name: str | None = None,
    ):
        search_args = {
            "collection_name": self.collection_name,
            "query": query_vector,
            "limit": top_k,
            "with_payload": True,
        }

        if source_name:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            search_args["query_filter"] = Filter(
                must=[
                    FieldCondition(
                        key="metadata.source",
                        match=MatchValue(value=source_name),
                    )
                ]
            )

        results = self.client.query_points(**search_args)
        return results.points

    def get_all_documents(self):
        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=1000,
            with_payload=True,
        )

        return [
            {
                "text": record.payload["text"],
                "metadata": record.payload["metadata"],
            }
            for record in records
        ]