"""
Хранилище на Milvus Lite: локальный файл БД, вставка и поиск через MilvusClient.
"""

import json
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from pathlib import Path
from os import getenv

import numpy as np
from pymilvus import MilvusClient


@dataclass
class MilvusConfig:
    db_path: str
    collection: str
    metric: str
    topk_limit: int

    @staticmethod
    def load() -> "MilvusConfig":
        return MilvusConfig(
            db_path=getenv("MILVUS_DB_PATH", "./data/milvus.db"),
            collection=getenv("MILVUS_COLLECTION", "rag_chunks"),
            metric=(getenv("MILVUS_METRIC", "COSINE") or "COSINE").upper(),
            topk_limit=int(getenv("MILVUS_TOPK_LIMIT", "200")),
        )


def _id64(s: str) -> int:
    """
    Детерминированный int64 из строки.
    """
    d = hashlib.sha1(s.encode("utf-8"), usedforsecurity=False).digest()
    v = int.from_bytes(d[:8], "big", signed=False)
    return v & ((1 << 63) - 1)


class MilvusStore:
    """
    Обёртка над Milvus Lite для операций upsert и поиска чанков.
    """

    def __init__(
        self,
        dim: int,
        cfg: MilvusConfig | None = None,
        collection: str | None = None,
    ) -> None:
        self.cfg = cfg or MilvusConfig.load()
        if collection:
            self.cfg = MilvusConfig(
                db_path=self.cfg.db_path,
                collection=collection,
                metric=self.cfg.metric,
                topk_limit=self.cfg.topk_limit,
            )
        Path(self.cfg.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.client = MilvusClient(self.cfg.db_path)
        if not self.client.has_collection(self.cfg.collection):
            self.client.create_collection(
                collection_name=self.cfg.collection,
                dimension=dim,
                metric_type=self.cfg.metric,
            )

    def upsert_chunks(
        self,
        doc_id: str,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        vectors: np.ndarray,
    ) -> int:
        """
        Вставляет или обновляет чанки документа. Возвращает количество записей.
        """
        data = []
        for i, (t, m, v) in enumerate(zip(texts, metadatas, vectors)):
            rid = _id64(f"{doc_id}::{i:06d}")
            payload = {"doc_id": doc_id, "text": t, **m}
            payload = json.loads(json.dumps(payload))
            data.append(
                {
                    "id": int(rid),
                    "vector": np.asarray(v, dtype=np.float32).tolist(),
                    **payload,
                }
            )
        self.client.upsert(collection_name=self.cfg.collection, data=data)
        return len(data)

    def search(self, query_vec: np.ndarray, top_k: int) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Ищет ближайшие чанки и возвращает пары (score, payload).
        """
        k = min(int(top_k), self.cfg.topk_limit)
        res = self.client.search(
            collection_name=self.cfg.collection,
            data=[np.asarray(query_vec, dtype=np.float32).tolist()],
            limit=k,
            output_fields=["doc_id", "text", "chunk_id", "source_path"],
        )
        hits = res[0] if res else []
        out: List[Tuple[float, Dict[str, Any]]] = []
        for h in hits:
            score = float(h.get("distance", 0.0))
            payload: Dict[str, Any] = {k: v for k, v in h.items() if k not in {
                "id", "distance", "entity"}}
            ent = h.get("entity")
            if ent is not None:
                if isinstance(ent, dict):
                    for key in ("doc_id", "text", "chunk_id", "source_path"):
                        if key in ent and key not in payload:
                            payload[key] = ent.get(key)
                else:
                    try:
                        for key in ("doc_id", "text", "chunk_id", "source_path"):
                            # объект hit.entity в ORM имеет .get()
                            val = ent.get(key)
                            if val is not None and key not in payload:
                                payload[key] = val
                    except Exception:
                        pass
            out.append((score, payload))
        return out
