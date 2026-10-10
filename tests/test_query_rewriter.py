from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.retrieval.query_rewriter import QueryRewriter


@pytest.fixture
def rewriter(monkeypatch):
    monkeypatch.setattr(
        "app.retrieval.query_rewriter.load_dotenv",
        lambda: None,
    )
    monkeypatch.setenv("GROQ_API_KEY", "test-api-key")

    mock_groq = Mock()
    monkeypatch.setattr(
        "app.retrieval.query_rewriter.Groq",
        lambda **kwargs: mock_groq,
    )

    return QueryRewriter(), mock_groq


def mock_response(content):
    return SimpleNamespace(
        model="openai/gpt-oss-20b",
        usage=None,
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    content=content,
                    reasoning=None,
                ),
            )
        ],
    )


def test_rewrites_query_and_strips_whitespace(rewriter):
    instance, mock_groq = rewriter
    mock_groq.chat.completions.create.return_value = mock_response(
        "  BM25 keyword-based retrieval in RAG systems  "
    )

    result = instance.rewrite(
        "How does BM25 improve retrieval in a RAG system?"
    )

    assert result == "BM25 keyword-based retrieval in RAG systems"
    mock_groq.chat.completions.create.assert_called_once()


def test_empty_response_returns_original_query(rewriter):
    instance, mock_groq = rewriter
    mock_groq.chat.completions.create.return_value = mock_response("  ")
    question = "How does BM25 improve retrieval?"

    assert instance.rewrite(question) == question


def test_api_failure_returns_original_query(rewriter):
    instance, mock_groq = rewriter
    mock_groq.chat.completions.create.side_effect = RuntimeError(
        "Simulated API failure"
    )
    question = "How does BM25 improve retrieval?"

    assert instance.rewrite(question) == question


def test_missing_api_key_raises_error(monkeypatch):
    monkeypatch.setattr(
        "app.retrieval.query_rewriter.load_dotenv",
        lambda: None,
    )
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GROQ_API_KEY missing"):
        QueryRewriter()
