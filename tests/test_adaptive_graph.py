
from types import SimpleNamespace

import pytest

from app.routing.router import RouteDecision
from app.routing.service import AdaptiveRAGService


class FakeRouter:
    def __init__(self, route):
        self.route = route

    def classify(self, question):
        return RouteDecision(
            route=self.route,
            reason=f"Mocked route: {self.route}",
        )


def make_service(route):
    service = AdaptiveRAGService.__new__(AdaptiveRAGService)

    service.router = FakeRouter(route)
    service.rag_service = SimpleNamespace(
        query=lambda question, top_k: {
            "answer": "Answer from document RAG",
            "sources": [{"document": "test.pdf"}],
        }
    )

    service._generate_direct = lambda question: "General answer"

    service._search_web = lambda question: [
        {
            "title": "Test source",
            "url": "https://example.com",
            "snippet": "Test evidence",
        }
    ]

    service._answer_from_web = (
        lambda question, sources: "Web-grounded answer"
    )

    service.graph = service._build_graph()
    return service


@pytest.mark.parametrize(
    ("route", "expected_answer", "expected_sources"),
    [
        (
            "document_rag",
            "Answer from document RAG",
            [{"document": "test.pdf"}],
        ),
        ("general_llm", "General answer", []),
        (
            "web_search",
            "Web-grounded answer",
            [
                {
                    "title": "Test source",
                    "url": "https://example.com",
                    "snippet": "Test evidence",
                }
            ],
        ),
    ],
)
def test_graph_selects_correct_route(
    route, expected_answer, expected_sources
):
    service = make_service(route)

    result = service.query("Explain this question", top_k=3)

    assert result["route"] == route
    assert result["answer"] == expected_answer
    assert result["sources"] == expected_sources
    assert result["route_reason"] == f"Mocked route: {route}"


def test_empty_question_is_rejected():
    service = make_service("general_llm")

    with pytest.raises(ValueError, match="Question cannot be empty"):
        service.query("   ")


@pytest.mark.parametrize("top_k", [0, 21])
def test_invalid_top_k_is_rejected(top_k):
    service = make_service("document_rag")

    with pytest.raises(ValueError, match="top_k must be between 1 and 20"):
        service.query("Find something", top_k=top_k)
