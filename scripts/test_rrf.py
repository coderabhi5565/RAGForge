import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.fusion import RRFFusion


vector_store = VectorStore()

documents = vector_store.get_all_documents()

bm25 = BM25Retriever(
    documents
)

query = "What are the possible booking statuses?"

bm25_results = bm25.retrieve(
    query=query,
    top_k=5,
)

dense_query = "What are the possible booking statuses?"

from app.retrieval.embeddings import EmbeddingService

embedding_service = EmbeddingService()

query_vector = embedding_service.embed_query(
    dense_query
)

dense_results = vector_store.similarity_search(
    query_vector=query_vector,
    top_k=5,
)

dense_results = [
    {
        "text": result.payload["text"],
        "metadata": result.payload["metadata"],
    }
    for result in dense_results
]

bm25_documents = [
    result[0]
    for result in bm25_results
]

fusion = RRFFusion()

results = fusion.fuse(
    rankings=[
        dense_results,
        bm25_documents,
    ],
    top_k=5,
)

print("\n===== RRF Fusion =====\n")

print(f"Query: {query}\n")

for rank, result in enumerate(
    results,
    start=1,
):
    document, score = result

    print(
        f"#{rank} "
        f"{document['metadata']['chunk_id']} "
        f"rrf_score={score:.6f}"
    )