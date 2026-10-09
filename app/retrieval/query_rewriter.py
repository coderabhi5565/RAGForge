
import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()


class QueryRewriter:
    def __init__(self, model: str = "openai/gpt-oss-20b"):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY missing. Add it to your .env file."
            )

        self.client = Groq(api_key=api_key)
        self.model = model

    def rewrite(self, question: str) -> str:
        prompt = f"""
Rewrite the user's question into a concise query for document retrieval.

Rules:
- Preserve the original meaning and intent.
- Keep important entities and technical terms.
- Do not answer the question.
- Do not introduce unsupported facts.
- Return only the rewritten query.
- If the question is already suitable for retrieval, return it unchanged.

Original question:
{question}
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a query rewriting component for a "
                        "Retrieval-Augmented Generation system. "
                        "Return only the rewritten query."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
            max_completion_tokens=256,
            include_reasoning=False,
        )

        rewritten_query = response.choices[0].message.content

        if not rewritten_query or not rewritten_query.strip():
            raise RuntimeError(
                "Groq returned empty output during query rewriting."
            )

        return rewritten_query.strip()
