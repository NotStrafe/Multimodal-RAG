"""
Поиск релевантных фрагментов и формирование текстового ответа без LLM.
"""

from typing import List, Dict, Tuple
from collections import defaultdict

import numpy as np

from .embeddings import embed_texts, embedding_dim
from .store import MilvusStore


def _group_top_by_doc(results: List[Tuple[float, Dict]], top_docs: int, chunks_per_doc: int) -> List[Dict]:
    by_doc: Dict[str, List[Tuple[float, Dict]]] = defaultdict(list)
    for score, payload in results:
        by_doc[payload.get("doc_id", "")].append((score, payload))
    ordered_docs = sorted(by_doc.items(), key=lambda kv: min(
        s for s, _ in kv[1]))[:top_docs]
    out = []
    for doc_id, items in ordered_docs:
        items_sorted = sorted(items, key=lambda t: t[0])[:chunks_per_doc]
        out.append({"doc_id": doc_id, "chunks": [p for _, p in items_sorted]})
    return out


def answer_with_top_docs(query: str, top_docs: int = 5, chunks_per_doc: int = 3) -> str:
    """
    Возвращает текст с лучшими фрагментами по запросу без участия LLM.
    """
    dim = embedding_dim()
    store = MilvusStore(dim=dim)
    qv = embed_texts([query])[0]
    hits = store.search(qv, top_k=top_docs * max(chunks_per_doc, 1))
    grouped = _group_top_by_doc(
        hits, top_docs=top_docs, chunks_per_doc=chunks_per_doc)
    lines: List[str] = []
    for g in grouped:
        lines.append(f"[{g['doc_id']}]")
        for ch in g["chunks"]:
            txt = str(ch.get("text", "")).strip()
            if len(txt) > 800:
                txt = txt[:800] + " …"
            lines.append(f"— {txt}")
        lines.append("")
    if not lines:
        return "Ничего релевантного не найдено."
    return "\n".join(lines).strip()
