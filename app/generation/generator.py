
from dotenv import load_dotenv
from groq import Groq


load_dotenv()


class Generator:
    def __init__(self, model: str = "openai/gpt-oss-20b"):
        self.client = Groq()
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
                        "Do not invent facts."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_completion_tokens=2048,
        )

        answer = response.choices[0].message.content

        if not answer:
            raise RuntimeError("Groq returned an empty answer.")

        return answer.strip()
