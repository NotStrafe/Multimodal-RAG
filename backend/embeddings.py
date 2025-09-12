"""
Загрузка и использование Jina Embeddings v3 через transformers с локальным кэшем и ретраем.
"""

from os import getenv
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


_MODEL_NAME = getenv(
    "EMBED_MODEL_NAME", "jinaai/jina-embeddings-v3").strip() or "jinaai/jina-embeddings-v3"
_CACHE_DIR = Path(getenv("HF_CACHE_DIR", "./.hf_cache")).resolve()


def _purge_dyn_modules() -> None:
    """
    Удаляет кэш динамических модулей для моделей Jina.
    """
    home = Path.home()
    p1 = home / ".cache" / "huggingface" / \
        "modules" / "transformers_modules" / "jinaai"
    p2 = _CACHE_DIR / "modules" / "transformers_modules" / "jinaai"
    for p in (p1, p2):
        if p.exists():
            for child in p.rglob("*"):
                if child.is_file():
                    try:
                        child.unlink()
                    except Exception:
                        pass
            try:
                for sub in sorted(p.rglob("*"), reverse=True):
                    if sub.is_dir():
                        sub.rmdir()
                p.rmdir()
            except Exception:
                pass


def _load_models() -> Tuple[AutoTokenizer, AutoModel]:
    """
    Загружает токенайзер и модель, при ошибке FileNotFound повторяет после очистки кэша.
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        tok = AutoTokenizer.from_pretrained(
            _MODEL_NAME, trust_remote_code=True, cache_dir=str(_CACHE_DIR))
        mdl = AutoModel.from_pretrained(
            _MODEL_NAME, trust_remote_code=True, cache_dir=str(_CACHE_DIR))
        return tok, mdl
    except FileNotFoundError:
        _purge_dyn_modules()
        tok = AutoTokenizer.from_pretrained(
            _MODEL_NAME, trust_remote_code=True, cache_dir=str(_CACHE_DIR), force_download=True)
        mdl = AutoModel.from_pretrained(
            _MODEL_NAME, trust_remote_code=True, cache_dir=str(_CACHE_DIR), force_download=True)
        return tok, mdl


_tokenizer, _model = _load_models()
_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
_model.to(_device)
_model.eval()


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Среднее по токенам с маской внимания.
    """
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Преобразует список строк в матрицу L2-нормированных эмбеддингов [N, D] (float32).
    """
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            toks = _tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            toks = {k: v.to(_device) for k, v in toks.items()}
            outputs = _model(**toks)
            pooled = _mean_pool(outputs.last_hidden_state,
                                toks["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            out.append(pooled.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0) if out else np.zeros((0, _model.config.hidden_size), dtype=np.float32)


def embedding_dim() -> int:
    """
    Возвращает размерность эмбеддинга.
    """
    v = embed_texts(["_probe_"])
    return int(v.shape[1])
