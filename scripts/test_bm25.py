import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25 import BM25Retriever


vector_store = VectorStore()

documents = vector_store.get_all_documents()

bm25 = BM25Retriever(
    documents
)

query = "What are the possible booking statuses?"

results = bm25.retrieve(
    query=query,
    top_k=5,
)

print("\n===== BM25 Retrieval =====\n")

print(f"Query: {query}\n")

for rank, result in enumerate(
    results,
    start=1,
):
    document, score = result

    print(
        f"#{rank} "
        f"{document['metadata']['chunk_id']} "
        f"score={score:.4f}"
    )

    print(
        document["text"][:300]
    )

    print()