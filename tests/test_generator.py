
from unittest.mock import Mock

from app.generation.generator import Generator


def make_generator():
    generator = Generator.__new__(Generator)
    generator.client = Mock()
    generator.model = "test-model"
    return generator


def mock_response(content):
    response = Mock()
    response.choices[0].message.content = content
    return response


def test_returns_answer_when_context_supports_question():
    generator = make_generator()
    generator.client.chat.completions.create.return_value = (
        mock_response(
            '{"answerable": true, "answer": "Razorpay"}'
        )
    )

    answer = generator.generate(
        query="What payment gateway does Foodingo use?",
        context=["Foodingo integrates Razorpay payment processing."],
    )

    assert answer == "Razorpay"


def test_abstains_when_context_is_insufficient():
    generator = make_generator()
    generator.client.chat.completions.create.return_value = (
        mock_response(
            '{"answerable": false, "answer": '
            '"I don\'t have enough information in the provided documents."}'
        )
    )

    answer = generator.generate(
        query="What is the candidate's date of birth?",
        context=["The candidate studies at OIST Bhopal."],
    )

    assert answer == Generator.FALLBACK_ANSWER


def test_empty_context_abstains_without_api_call():
    generator = make_generator()

    answer = generator.generate(
        query="What is MCP?",
        context=[],
    )

    assert answer == Generator.FALLBACK_ANSWER
    generator.client.chat.completions.create.assert_not_called()


def test_invalid_json_raises_error():
    generator = make_generator()
    generator.client.chat.completions.create.return_value = (
        mock_response("This is not JSON")
    )

    try:
        generator.generate(
            query="What is BM25?",
            context=["BM25 is a lexical retrieval algorithm."],
        )
        assert False, "Expected RuntimeError"
    except RuntimeError as error:
        assert "invalid JSON" in str(error)
