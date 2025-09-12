"""
Поиск релевантных фрагментов и RAG-ответ через GigaChat.
"""

from typing import List, Dict, Tuple
from collections import defaultdict

from .embeddings import embed_texts, embedding_dim
from .store import MilvusStore
from . import llm_gigachat


def _group_top_by_doc(results: List[Tuple[float, Dict]], top_docs: int, chunks_per_doc: int) -> List[Dict]:
    by_doc = defaultdict(list)
    for score, payload in results:
        by_doc[payload.get("doc_id", "")].append((score, payload))
    ordered = sorted(by_doc.items(), key=lambda kv: min(
        s for s, _ in kv[1]))[:top_docs]
    out = []
    for doc_id, items in ordered:
        best = sorted(items, key=lambda t: t[0])[:chunks_per_doc]
        out.append({"doc_id": doc_id, "chunks": [p for _, p in best]})
    return out


def answer_with_top_docs(query: str, top_docs: int = 5, chunks_per_doc: int = 3) -> str:
    """
    Возвращает текст с лучшими фрагментами без участия LLM.
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


def answer_rag_with_llm(query: str, top_docs: int = 4, chunks_per_doc: int = 3, max_ctx_chars: int = 12000) -> str:
    """
    Возвращает финальный ответ LLM на основе найденных фрагментов.
    """
    dim = embedding_dim()
    store = MilvusStore(dim=dim)
    qv = embed_texts([query])[0]
    hits = store.search(qv, top_k=top_docs * max(chunks_per_doc, 1))
    grouped = _group_top_by_doc(
        hits, top_docs=top_docs, chunks_per_doc=chunks_per_doc)

    if not grouped:
        system = "Отвечай кратко и честно. Если информации нет, скажи, что ответ в документах не найден."
        user = f"Вопрос: {query}\n\nКонтекст отсутствует."
        return llm_gigachat.generate(system, user)

    ctx_lines: List[str] = []
    for g in grouped:
        ctx_lines.append(f"[DOC {g['doc_id']}]")
        for ch in g["chunks"]:
            txt = str(ch.get("text", "")).strip().replace("\n", " ")
            if len(txt) > 800:
                txt = txt[:800] + " …"
            ctx_lines.append(f"- {txt}")
    ctx_text = "\n".join(ctx_lines)
    if len(ctx_text) > max_ctx_chars:
        ctx_text = ctx_text[:max_ctx_chars] + " …"

    system = (
        "Ты отвечаешь строго на основе предоставленного контекста из технической документации. "
        "Если контекст не содержит ответа, прямо скажи об этом и предложи переформулировать вопрос. "
        "Форматируй кратко и по делу."
    )
    user = (
        f"Вопрос: {query}\n\n"
        f"Контекст:\n{ctx_text}\n\n"
        "Сформулируй ответ по контексту. В конце добавь строку «Источники:» с перечислением DOC-идентификаторов в квадратных скобках."
    )
    return llm_gigachat.generate(system, user)
