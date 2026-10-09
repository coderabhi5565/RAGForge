
from app.generation.generator import Generator


def main():
    generator = Generator()

    query = "What is BM25?"

    context = [
        (
            "BM25 is a lexical information retrieval algorithm "
            "used to rank documents based on their relevance to "
            "a search query. It considers term frequency, inverse "
            "document frequency, and document length."
        )
    ]

    print("Question:", query)
    print("\nGenerating answer using Groq...\n")

    answer = generator.generate(
        query=query,
        context=context,
    )

    print("Answer:")
    print(answer)


if __name__ == "__main__":
    main()
