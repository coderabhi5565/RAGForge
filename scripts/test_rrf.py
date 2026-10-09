import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.fusion import RRFFusion
from app.retrieval.embeddings import EmbeddingService


vector_store = VectorStore()
documents = vector_store.get_all_documents()
bm25 = BM25Retriever(documents)

query = "What are the possible booking statuses?"

bm25_results = bm25.retrieve(query=query, top_k=5)

embedding_service = EmbeddingService()
query_vector = embedding_service.embed_query(query)

dense_results = vector_store.similarity_search(
    query_vector=query_vector,
    top_k=5,
)

dense_documents = [
    {
        "text": result.payload["text"],
        "metadata": result.payload["metadata"],
    }
    for result in dense_results
    if result.payload
    and result.payload.get("text")
    and result.payload.get("metadata") is not None
]

fusion = RRFFusion()

results = fusion.fuse(
    rankings=[dense_documents, bm25_results],
    top_k=5,
)

print("\n===== RRF Fusion =====\n")
print(f"Query: {query}")
print(f"Dense results: {len(dense_documents)}")
print(f"BM25 results: {len(bm25_results)}")
print(f"Fused results: {len(results)}\n")

for rank, document in enumerate(results, start=1):
    print(
        f"#{rank} "
        f"{document['metadata'].get('chunk_id', 'unknown_chunk')}"
    )
    print(document["text"][:300])
    print()
