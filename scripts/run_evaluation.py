import json
import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from evaluation.evaluator.runner import EvaluationRunner


DATASET_PATH = ROOT_DIR / "evaluation" / "dataset" / "rag_eval.json"
RESULTS_DIR = ROOT_DIR / "evaluation" / "results"
REPORT_PATH = RESULTS_DIR / "baseline_hybrid_reranking.json"


def main():
    runner = EvaluationRunner()

    results = runner.run(str(DATASET_PATH))
    metrics = runner.calculate_metrics(results)

    report = {
        "experiment": "baseline_hybrid_reranking",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(DATASET_PATH.relative_to(ROOT_DIR)),
        "query_count": len(results),
        "metrics": metrics,
        "per_query_results": results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\n===== RAGForge Retrieval Baseline =====\n")

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print(f"\nQueries evaluated: {len(results)}")
    print(f"Report saved: {REPORT_PATH}")
    print("\n=======================================\n")


if __name__ == "__main__":
    main()