import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from app.retrieval.vector_store import VectorStore

from app.retrieval.vector_store import VectorStore


def main():

    vector_store = VectorStore()

    results = vector_store.client.scroll(
        collection_name=vector_store.collection_name,
        limit=100,
        with_payload=True,
    )

    points = results[0]

    print(f"Total chunks fetched: {len(points)}")
    print("=" * 100)

    for point in points:

        metadata = point.payload["metadata"]

        print(f"Qdrant Point ID: {point.id}")
        print(f"Chunk ID: {metadata.get('chunk_id')}")
        print(f"Source: {metadata.get('source')}")
        print(f"Document ID: {metadata.get('document_id')}")
        print(f"Metadata: {metadata}")

        print("\nText:")
        print(point.payload["text"])

        print("=" * 100)


if __name__ == "__main__":
    main()