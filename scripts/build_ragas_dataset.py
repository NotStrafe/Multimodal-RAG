"""
Сбор датасета для Ragas на основе индексированного корпуса.

Исходник вопросов/эталонных ответов: data/ragas_eval_seed.jsonl
Результат с ответами и фактическим контекстом: data/ragas_eval_generated.jsonl
"""

from __future__ import annotations

import json
import sys
from os import getenv
from pathlib import Path
from typing import List, Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.store import MilvusStore  # noqa: E402
from backend.clip_encoder import embed_queries, clip_dim  # noqa: E402
from backend.embeddings import embed_texts  # noqa: E402
from backend.rag_qa import (  # noqa: E402
    answer_rag_with_llm,
    answer_with_top_docs,
    _group_top_by_doc,
)


def _fetch_contexts(
    query: str,
    top_docs: int,
    chunks_per_doc: int,
    img_top: int,
) -> List[str]:
    """Возвращает объединённый список текстовых и OCR-контекстов, найденных системой."""
    # Текстовые чанки
    dim = embed_texts(["_probe_"])[0].shape[0]
    store = MilvusStore(dim=dim)
    qv = embed_texts([query])[0]
    hits = store.search(qv, top_k=top_docs * max(chunks_per_doc, 1))
    grouped = _group_top_by_doc(hits, top_docs=top_docs,
                                chunks_per_doc=chunks_per_doc)
    ctx: List[str] = []
    for g in grouped:
        for ch in g["chunks"]:
            txt = str(ch.get("text", "")).strip()
            if txt:
                ctx.append(txt)

    # Визуальные (OCR-текст, сохранённый при индексации изображений)
    img_store = MilvusStore(
        dim=clip_dim(), collection=getenv("MILVUS_IMAGE_COLLECTION", "rag_images")
    )
    img_vec = embed_queries([query])[0]
    img_hits = img_store.search(img_vec, top_k=img_top)
    for _, payload in img_hits:
        txt = str(payload.get("text", "")).strip()
        if txt:
            ctx.append(txt)
    return ctx


def main() -> None:
    seed_path = Path("data/ragas_eval_seed.jsonl")
    out_path = Path("data/ragas_eval_generated.jsonl")
    if not seed_path.exists():
        raise FileNotFoundError(seed_path)

    top_docs = int(getenv("TOP_DOCS", "5"))
    chunks_per_doc = int(getenv("CHUNKS_PER_DOC", "3"))
    img_top = int(getenv("TOP_IMAGES", "3"))

    use_llm = (getenv("RAG_USE_LLM") or "true").strip().lower() in {
        "1", "true", "yes", "y", "t"}

    entries: List[Dict] = []
    for line in seed_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        question = item["question"]
        ground_truth = item.get("ground_truth", "")

        contexts = _fetch_contexts(
            question, top_docs=top_docs, chunks_per_doc=chunks_per_doc, img_top=img_top
        )
        if use_llm:
            answer = answer_rag_with_llm(
                question, top_docs=top_docs, chunks_per_doc=chunks_per_doc
            )
        else:
            answer = answer_with_top_docs(
                question, top_docs=top_docs, chunks_per_doc=chunks_per_doc
            )
        entries.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in entries:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {len(entries)} rows to {out_path}")


if __name__ == "__main__":
    main()
