import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25 import BM25Retriever


def main():
    vector_store = VectorStore()
    documents = vector_store.get_all_documents()

    print("\n===== BM25 Retrieval Test =====\n")
    print(f"Documents loaded: {len(documents)}")

    if not documents:
        print("No documents found in Qdrant. Ingest a document first.")
        return

    bm25 = BM25Retriever(documents)

    query = "What are the possible booking statuses?"
    results = bm25.retrieve(query=query, top_k=5)

    print(f"\nQuery: {query}")
    print(f"Results returned: {len(results)}\n")

    if not results:
        print("No matching documents found.")
        return

    for rank, document in enumerate(results, start=1):
        metadata = document.get("metadata", {})
        chunk_id = metadata.get("chunk_id", "unknown_chunk")

        print(f"#{rank} {chunk_id}")
        print(document["text"][:300])
        print("-" * 60)


if __name__ == "__main__":
    main()
