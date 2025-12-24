"""
Запуск ragas-оценки на сгенерированном датасете.

Вход: data/ragas_eval_generated.jsonl (из build_ragas_dataset.py)
Метрики: context_precision, context_recall, faithfulness, answer_correctness
Выход: печать результата и сохранение CSV в data/ragas_eval_report.csv
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv, find_dotenv
from ragas import evaluate
from ragas.metrics.collections import (
    answer_correctness,
    context_precision,
    context_recall,
    faithfulness,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.llm_gigachat import _client as gigachat_client  # noqa: E402
from backend.embeddings import embed_texts  # noqa: E402


class LocalEmbeddings:
    """
    Адаптер эмбеддингов для ragas 0.1.x, использующий embed_texts.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = embed_texts(texts)
        return [v.astype(np.float32).tolist() for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        vec = embed_texts([text])[0]
        return vec.astype(np.float32).tolist()


def main() -> None:
    load_dotenv(find_dotenv(usecwd=True), override=False)

    data_path = Path("data/ragas_eval_generated.jsonl")
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} не найден. Сначала запустите scripts/build_ragas_dataset.py"
        )

    llm = gigachat_client()
    embeds = LocalEmbeddings()

    ds = load_dataset("json", data_files=str(data_path))["train"]

    metrics = [
        context_precision,
        context_recall,
        faithfulness,
        answer_correctness,
    ]

    result = evaluate(ds, metrics=metrics, llm=llm, embeddings=embeds)
    print("Ragas result:")
    print(result)

    df = result.to_pandas()
    out_csv = Path("data/ragas_eval_report.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved per-sample metrics to {out_csv}")


if __name__ == "__main__":
    main()
