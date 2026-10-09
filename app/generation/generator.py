import json
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

class Generator:
    FALLBACK_ANSWER = (
        "I don't have enough information in the provided documents."
    )

    def __init__(self, model: str = "openai/gpt-oss-20b"):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY missing. Add it to your .env file."
            )

        self.client = Groq(api_key=api_key)
        self.model = model

    def generate(self, query: str, context: list[str]) -> str:
        usable_context = [
            text.strip()
            for text in context
            if isinstance(text, str) and text.strip()
        ]

        if not usable_context:
            return self.FALLBACK_ANSWER

        context_text = "\n\n".join(
            f"[Context {index}]\n{text}"
            for index, text in enumerate(usable_context, start=1)
        )

        prompt = f"""
Determine whether the provided context contains enough evidence
to answer the question.

Rules:
- Use only facts explicitly supported by the context.
- Related keywords alone do not establish that an answer is supported.
- Do not use outside knowledge to fill missing information.
- Do not invent definitions, names, dates, figures, or implementation details.
- If only part of the question can be answered, answer only that part
  if it is useful; otherwise mark it unanswerable.
- If the context is insufficient, set "answerable" to false.
- If answerable is false, use this exact answer:
  "{self.FALLBACK_ANSWER}"
- Return valid JSON only, with keys "answerable" and "answer".
- "answerable" must be a boolean.
- "answer" must be a string.

Context:
{context_text}

Question:
{query}
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict, evidence-grounded RAG answerer. "
                        "Assess whether the supplied context supports the "
                        "answer. Never fill evidence gaps with prior knowledge. "
                        "Return only the requested JSON object."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_completion_tokens=2048,
            include_reasoning=False,
        )

        output = response.choices[0].message.content

        if not output or not output.strip():
            raise RuntimeError(
                "Groq returned empty output during answer generation."
            )

        try:
            result = json.loads(output.strip())
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Groq returned invalid JSON during answer generation."
            ) from exc

        if (
            not isinstance(result, dict)
            or not isinstance(result.get("answerable"), bool)
            or not isinstance(result.get("answer"), str)
            or not result["answer"].strip()
        ):
            raise RuntimeError(
                "Groq returned an invalid answerability response."
            )

        if not result["answerable"]:
            return self.FALLBACK_ANSWER

        return result["answer"].strip()
