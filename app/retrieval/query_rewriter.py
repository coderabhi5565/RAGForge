
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

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a query rewriting component for a "
                            "Retrieval-Augmented Generation system. "
                            "Return only the rewritten query, without "
                            "reasoning or explanation."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0,
                max_completion_tokens=512,
                include_reasoning=False,
            )

            choice = response.choices[0]
            message = choice.message

            # Diagnostic information for debugging empty responses
            print("\n===== GROQ QUERY REWRITE DIAGNOSTICS =====")
            print("Model:", response.model)
            print("Finish reason:", choice.finish_reason)
            print("Usage:", response.usage)
            print("Message content:", repr(message.content))
            print(
                "Reasoning:",
                repr(getattr(message, "reasoning", None)),
            )
            print("==========================================\n")

            rewritten_query = message.content

            if rewritten_query and rewritten_query.strip():
                return rewritten_query.strip()

            print(
                "[QueryRewriter] Empty response from Groq; "
                "using original query."
            )
            return question

        except Exception as exc:
            print(
                f"[QueryRewriter] Groq request failed: {exc}. "
                "Using original query."
            )
            return question
