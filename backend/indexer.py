"""
Индексация файла: парсинг, чанкинг, эмбеддинги и запись в Milvus-lite.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
from os import getenv
from datetime import datetime

import numpy as np
from pypdf import PdfReader
from docx import Document as DocxDocument
from bs4 import BeautifulSoup

from .embeddings import embed_texts, embedding_dim
from .store import MilvusStore


@dataclass
class IngestConfig:
    chunk_tokens: int
    chunk_overlap: int

    @staticmethod
    def load() -> "IngestConfig":
        return IngestConfig(
            chunk_tokens=int(getenv("CHUNK_SIZE_TOKENS", "800")),
            chunk_overlap=int(getenv("CHUNK_OVERLAP_TOKENS", "120")),
        )


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)


def _read_docx(path: Path) -> str:
    d = DocxDocument(str(path))
    return "\n".join(p.text for p in d.paragraphs)


def _read_html(path: Path) -> str:
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(" ", strip=True)


def _split_tokens(text: str, max_tokens: int, overlap: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    step = max_tokens - overlap
    while i < len(words):
        chunk = words[i: i + max_tokens]
        chunks.append(" ".join(chunk))
        i += max(step, 1)
    return chunks


def _parse_file(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    if ext in {"txt", "md"}:
        return _read_txt(path)
    if ext == "pdf":
        return _read_pdf(path)
    if ext == "docx":
        return _read_docx(path)
    if ext == "html":
        return _read_html(path)
    raise ValueError(f"Неизвестный формат: .{ext}")


def index_file(path_str: str) -> Tuple[str, int]:
    """
    Индексирует файл в хранилище. Возвращает doc_id и число чанков.
    """
    cfg = IngestConfig.load()
    path = Path(path_str).resolve()
    if not path.exists():
        raise FileNotFoundError(str(path))
    raw = _parse_file(path)
    if not raw.strip():
        raise ValueError("Файл пуст или не распознан")
    chunks = _split_tokens(raw, cfg.chunk_tokens, cfg.chunk_overlap)
    if not chunks:
        raise ValueError("Не удалось выделить чанки")
    vecs = embed_texts(chunks)
    dim = int(vecs.shape[1])
    store = MilvusStore(dim=dim)
    doc_id = f"{path.stem}__{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    metas = [{"source_path": str(path), "chunk_id": i}
             for i in range(len(chunks))]
    n = store.upsert_chunks(doc_id=doc_id, texts=chunks,
                            metadatas=metas, vectors=vecs)
    return doc_id, n
