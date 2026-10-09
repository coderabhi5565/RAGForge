
import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()


class Generator:
    def __init__(self, model: str = "openai/gpt-oss-20b"):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY missing. Add it to your .env file."
            )

        self.client = Groq(api_key=api_key)
        self.model = model

    def generate(self, query: str, context: list[str]) -> str:
        context_text = "\n\n".join(context)

        prompt = f"""
You are a question-answering assistant.

Answer the user's question using only the provided context.

If the answer cannot be found in the context, say:
"I don't have enough information in the provided documents."

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
                        "Answer accurately using only the supplied context. "
                        "Do not invent facts. If the context does not contain "
                        "the answer, state that there is insufficient "
                        "information in the provided documents."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
            max_completion_tokens=2048,
            include_reasoning=False,
        )

        answer = response.choices[0].message.content

        if not answer or not answer.strip():
            raise RuntimeError(
                "Groq returned empty output during answer generation."
            )

        return answer.strip()
