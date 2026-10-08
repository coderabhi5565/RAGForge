import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from evaluation.evaluator.runner import EvaluationRunner


DATASET_PATH = "evaluation/dataset/rag_eval.json"


runner = EvaluationRunner()

results = runner.run(DATASET_PATH)

metrics = runner.calculate_metrics(results)

print("\n===== RAGForge Retrieval Evaluation =====\n")

for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

print("\n==========================================\n")