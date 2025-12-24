"""
Запуск ragas-оценки на сгенерированном датасете.

Вход: data/ragas_eval_generated.jsonl (из build_ragas_dataset.py)
Метрики: context_precision, context_recall, faithfulness, answer_correctness
Выход: печать результата и сохранение CSV в data/ragas_eval_report.csv
"""

from __future__ import annotations

from pathlib import Path

from datasets import load_dataset
from ragas import evaluate
from ragas.metrics.collections import (
    answer_correctness,
    context_precision,
    context_recall,
    faithfulness,
)


def main() -> None:
    data_path = Path("data/ragas_eval_generated.jsonl")
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} не найден. Сначала запустите scripts/build_ragas_dataset.py"
        )

    ds = load_dataset("json", data_files=str(data_path))["train"]

    metrics = [
        context_precision,
        context_recall,
        faithfulness,
        answer_correctness,
    ]

    result = evaluate(ds, metrics=metrics)
    print("Ragas result:")
    print(result)

    df = result.to_pandas()
    out_csv = Path("data/ragas_eval_report.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved per-sample metrics to {out_csv}")


if __name__ == "__main__":
    main()
