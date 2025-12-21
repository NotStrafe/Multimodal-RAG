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

from .embeddings import embed_texts
from .clip_encoder import embed_images, clip_dim
from .ocr import ocr_image
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

    @property
    def image_chunk_tokens(self) -> int:
        return max(self.chunk_tokens // 2, 50)


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


_IMG_EXT = {"png", "jpg", "jpeg", "webp"}


def _index_image(path: Path, cfg: IngestConfig) -> Tuple[str, int]:
    """
    Индексирует изображение: OCR -> текстовые чанки + CLIP вектор.
    """
    doc_id = f"{path.stem}__{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    ocr_text = ocr_image(path)
    chunk_texts = _split_tokens(
        ocr_text, cfg.image_chunk_tokens, cfg.chunk_overlap) if ocr_text else []

    total = 0
    # Текстовый индекс по OCR
    if chunk_texts:
        vecs = embed_texts(chunk_texts)
        metas = [{"source_path": str(path), "chunk_id": i,
                  "media_type": "image"} for i in range(len(chunk_texts))]
        store = MilvusStore(dim=int(vecs.shape[1]))
        store.upsert_chunks(doc_id=doc_id, texts=chunk_texts,
                            metadatas=metas, vectors=vecs)
        total += len(chunk_texts)

    # CLIP-вектор изображения
    clip_vec = embed_images([path])
    clip_store = MilvusStore(
        dim=clip_dim(), collection=getenv("MILVUS_IMAGE_COLLECTION", "rag_images"))
    clip_store.upsert_chunks(
        doc_id=doc_id,
        texts=[ocr_text or "[image]"],
        metadatas=[{"source_path": str(path), "chunk_id": 0,
                    "media_type": "image"}],
        vectors=clip_vec,
    )
    total += 1
    return doc_id, total


def index_file(path_str: str) -> Tuple[str, int]:
    """
    Индексирует файл в хранилище. Возвращает doc_id и число чанков.
    """
    cfg = IngestConfig.load()
    path = Path(path_str).resolve()
    if not path.exists():
        raise FileNotFoundError(str(path))
    ext = path.suffix.lower().lstrip(".")
    if ext in _IMG_EXT:
        return _index_image(path, cfg)
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
    metas = [{"source_path": str(path), "chunk_id": i, "media_type": "text"}
             for i in range(len(chunks))]
    n = store.upsert_chunks(doc_id=doc_id, texts=chunks,
                            metadatas=metas, vectors=vecs)
    return doc_id, n
