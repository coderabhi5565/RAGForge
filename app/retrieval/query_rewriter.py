
from app.generation.generator import Generator


class QueryRewriter:

    def __init__(self):
        self.generator = Generator()

    def rewrite(self, question: str) -> str:
        prompt = f"""
You are a query rewriting component for a Retrieval-Augmented
Generation (RAG) system.

Rewrite the user's question into a concise, retrieval-friendly
search query.

Rules:
- Preserve the original meaning and intent.
- Preserve important entities, technical terms, and constraints.
- Do not answer the question.
- Do not add facts that are not present in the question.
- Return only the rewritten query, without explanations.
- If the original query is already suitable for retrieval,
  return it unchanged.

Original question:
{question}
"""

        rewritten_query = self.generator.generate(
            query=prompt,
            context=[],
        )

        rewritten_query = rewritten_query.strip()

        if not rewritten_query:
            return question

        return rewritten_query
