from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()


class Generator:

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
    ):
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0,
        )

    def generate(
        self,
        query: str,
        context: list[str],
    ) -> str:

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

        response = self.llm.invoke(
            [HumanMessage(content=prompt)]
        )

        return response.content