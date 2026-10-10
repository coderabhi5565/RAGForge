import io
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api import routes


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_query_returns_answer_and_sources(monkeypatch):
    monkeypatch.setattr(
        routes.rag_service,
        "query",
        lambda question, top_k: {
            "answer": "Razorpay",
            "sources": [{"chunk_id": "resume_chunk_3"}],
        },
    )

    response = client.post(
        "/query",
        json={
            "question": "What payment gateway does Foodingo use?",
            "top_k": 5,
        },
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "Razorpay"
    assert response.json()["sources"]


@pytest.mark.parametrize("top_k", [0, -1, 21])
def test_query_rejects_invalid_top_k(top_k):
    response = client.post(
        "/query",
        json={"question": "What is BM25?", "top_k": top_k},
    )

    assert response.status_code == 422


def test_query_rejects_blank_question():
    response = client.post(
        "/query",
        json={"question": "   ", "top_k": 5},
    )

    assert response.status_code == 422


def test_upload_rejects_unsupported_file_type():
    response = client.post(
        "/documents/upload",
        files={"file": ("malware.exe", io.BytesIO(b"test"), "application/octet-stream")},
    )

    assert response.status_code == 415


def test_upload_rejects_empty_file():
    response = client.post(
        "/documents/upload",
        files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
    )

    assert response.status_code == 400
