
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.fusion import RRFFusion


def sample_document(chunk_id, text):
    return {
        "text": text,
        "metadata": {"chunk_id": chunk_id},
    }


def test_bm25_returns_relevant_document():
    documents = [
        sample_document(
            "chunk_1",
            "BM25 is a lexical retrieval algorithm for ranking search results",
        ),
        sample_document(
            "chunk_2",
            "Car rental booking and vehicle availability information",
        ),
        sample_document(
            "chunk_3",
            "BM25 retrieval ranks documents using lexical term matching",
        ),
    ]

    retriever = BM25Retriever(documents)
    results = retriever.retrieve("lexical retrieval algorithm", top_k=3)

    assert results
    assert results[0]["metadata"]["chunk_id"] in {"chunk_1", "chunk_3"}


def test_bm25_returns_empty_for_unmatched_query():
    documents = [
        sample_document("chunk_1", "Car rental and vehicle booking"),
    ]

    retriever = BM25Retriever(documents)
    results = retriever.retrieve("quantum physics", top_k=5)

    assert results == []


def test_rrf_fuses_rankings_without_duplicate_chunks():
    doc_a = sample_document("chunk_a", "BM25 retrieval")
    doc_b = sample_document("chunk_b", "Vector retrieval")
    doc_c = sample_document("chunk_c", "Hybrid retrieval")

    fusion = RRFFusion()
    results = fusion.fuse(
        rankings=[
            [doc_a, doc_b],
            [doc_a, doc_c],
        ],
        top_k=3,
    )

    chunk_ids = [doc["metadata"]["chunk_id"] for doc in results]

    assert chunk_ids[0] == "chunk_a"
    assert len(chunk_ids) == len(set(chunk_ids))
    assert len(results) == 3


def test_rrf_handles_empty_rankings():
    fusion = RRFFusion()

    assert fusion.fuse(rankings=[[], []], top_k=5) == []
