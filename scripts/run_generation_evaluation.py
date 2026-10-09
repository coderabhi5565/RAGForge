import sys
import re
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from groq import Groq
from app.retrieval.service import RAGService


load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "evaluation" / "dataset" / "rag_eval.json"
RESULTS_DIR = ROOT / "evaluation" / "results"
RESULTS_PATH = RESULTS_DIR / "generation_results.json"

MODEL = "openai/gpt-oss-20b"


def judge_answer(
    client: Groq,
    question: str,
    reference_answer: str,
    generated_answer: str,
    context: list[str],
) -> dict:
    context_text = "\n\n".join(context)

    prompt = f"""
Evaluate this RAG-generated answer using the supplied evidence.

QUESTION:
{question}

REFERENCE ANSWER:
{reference_answer}

RETRIEVED CONTEXT:
{context_text}

GENERATED ANSWER:
{generated_answer}

Give three integer scores from 1 to 5.

1. correctness:
How accurately does the generated answer address the question
compared with the reference answer?
Equivalent wording is acceptable. Do not require exact wording.

2. faithfulness:
How well is every factual claim in the generated answer supported
by the retrieved context?
Penalize unsupported or contradictory claims.

3. context_relevance:
How relevant is the retrieved context to answering the question?

Scoring:
1 = very poor
2 = poor
3 = partially adequate
4 = good
5 = excellent

Return ONLY a valid JSON object in this exact structure:
{{
  "correctness": 1,
  "faithfulness": 1,
  "context_relevance": 1,
  "reason": "Brief explanation of the scores"
}}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict evaluator of RAG systems. "
                    "Judge only from the supplied question, reference, "
                    "context, and generated answer. Return valid JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_completion_tokens=512,
        include_reasoning=False,
    )

    content = response.choices[0].message.content

    if not content or not content.strip():
        raise RuntimeError("Groq returned an empty evaluation response.")

    match = re.search(r"\{.*\}", content, re.DOTALL)

    if not match:
        raise ValueError(
            f"Could not find JSON in judge response: {content}"
        )

    scores = json.loads(match.group(0))

    for key in ("correctness", "faithfulness", "context_relevance"):
        value = scores.get(key)

        if not isinstance(value, int) or not 1 <= value <= 5:
            raise ValueError(
                f"Invalid judge score for {key}: {value}"
            )

    if not isinstance(scores.get("reason"), str):
        raise ValueError("Judge response is missing its reason.")

    return scores


def main():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY missing from .env")

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not dataset:
        raise ValueError("Generation evaluation dataset is empty.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    client = Groq(api_key=api_key)

    # Use the original query for retrieval.
    # Query rewriting remains OFF for this evaluation.
    rag_service = RAGService(use_query_rewriting=False)

    results = []

    try:
        for index, item in enumerate(dataset, start=1):
            question = item["question"]
            reference_answer = item["ground_truth_answer"]

            print(
                f"\n[{index}/{len(dataset)}] Evaluating: {question}"
            )

            documents = rag_service.retrieve(
                question=question,
                top_k=5,
            )

            context = [
                document.payload["text"]
                for document in documents
            ]

            generated_answer = rag_service.generator.generate(
                query=question,
                context=context,
            )

            scores = judge_answer(
                client=client,
                question=question,
                reference_answer=reference_answer,
                generated_answer=generated_answer,
                context=context,
            )

            result = {
                "id": item["id"],
                "question": question,
                "reference_answer": reference_answer,
                "generated_answer": generated_answer,
                "retrieved_chunk_ids": [
                    document.payload["metadata"]["chunk_id"]
                    for document in documents
                ],
                "scores": scores,
            }

            results.append(result)

            # Save after every successful query so progress is preserved.
            with open(RESULTS_PATH, "w", encoding="utf-8") as file:
                json.dump(results, file, indent=2, ensure_ascii=False)

            print(
                "Correctness:", scores["correctness"], "/ 5",
                "| Faithfulness:", scores["faithfulness"], "/ 5",
                "| Context relevance:", scores["context_relevance"], "/ 5",
            )

    finally:
        del rag_service

    total = len(results)

    summary = {
        "queries_evaluated": total,
        "average_correctness": round(
            sum(r["scores"]["correctness"] for r in results) / total, 3
        ),
        "average_faithfulness": round(
            sum(r["scores"]["faithfulness"] for r in results) / total, 3
        ),
        "average_context_relevance": round(
            sum(r["scores"]["context_relevance"] for r in results) / total, 3
        ),
    }

    output = {
        "model": MODEL,
        "evaluation_type": "LLM-as-a-judge",
        "score_range": "1-5",
        "summary": summary,
        "results": results,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print("\n===== GENERATION EVALUATION SUMMARY =====")
    print("Queries:", summary["queries_evaluated"])
    print("Average correctness:", summary["average_correctness"], "/ 5")
    print("Average faithfulness:", summary["average_faithfulness"], "/ 5")
    print(
        "Average context relevance:",
        summary["average_context_relevance"],
        "/ 5",
    )
    print("Saved results to:", RESULTS_PATH)


if __name__ == "__main__":
    main()
