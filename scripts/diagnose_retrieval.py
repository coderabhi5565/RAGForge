import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from evaluation.evaluator.runner import EvaluationRunner


DATASET_PATH = "evaluation/dataset/rag_eval.json"


def main():
    runner = EvaluationRunner()

    diagnostics = runner.run_diagnostics(
        DATASET_PATH,
        k=3,
    )

    print("\n===== Retrieval Diagnostics =====\n")

    for item in diagnostics:
        print(f"ID: {item['id']}")
        print(f"Question: {item['question']}")
        print(f"Relevant: {item['relevant_chunk_ids']}")
        print(f"Hit@1: {item['hit_at_1']}")

        for result in item["retrieved"]:
            rank = result.get("rank", "?")
            chunk_id = result.get("chunk_id", "unknown_chunk")
            score = result.get("score")

            if score is not None:
                print(
                    f"  #{rank} {chunk_id} "
                    f"score={score:.4f}"
                )
            else:
                print(f"  #{rank} {chunk_id}")

        print()


if __name__ == "__main__":
    main()
