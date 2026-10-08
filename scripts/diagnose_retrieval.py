import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from evaluation.evaluator.runner import EvaluationRunner


DATASET_PATH = "evaluation/dataset/rag_eval.json"


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
        print(
            f"  #{result['rank']} "
            f"{result['chunk_id']} "
            f"score={result['score']:.4f}"
        )

    print()