"""
CLIP-эмбеддинги Jina для изображений и текстовых запросов.
"""

from os import getenv
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

_MODEL_NAME = (getenv("CLIP_MODEL_NAME") or "jinaai/jina-clip-v2").strip()
_CACHE_DIR = Path(getenv("HF_CACHE_DIR", "./.hf_cache")).resolve()


def _load_clip() -> Tuple[AutoProcessor, AutoModel]:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(
        _MODEL_NAME, trust_remote_code=True, cache_dir=str(_CACHE_DIR)
    )
    model = AutoModel.from_pretrained(
        _MODEL_NAME, trust_remote_code=True, cache_dir=str(_CACHE_DIR)
    )
    return processor, model


_processor, _clip_model = _load_clip()
_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
_clip_model.to(_device)
_clip_model.eval()
_CLIP_DIM = int(
    getattr(_clip_model.config, "projection_dim", None)
    or getattr(_clip_model.config, "hidden_size", 0)
    or getattr(_clip_model.config, "text_config", {}).get("hidden_size", 0)
    or 512
)


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, p=2, dim=-1)


def embed_images(paths: List[Path], batch_size: int = 8) -> np.ndarray:
    """
    Возвращает L2-нормированные эмбеддинги изображений.
    """
    out: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch = paths[i: i + batch_size]
            imgs = [Image.open(p).convert("RGB") for p in batch]
            inputs = _processor(images=imgs, return_tensors="pt").to(_device)
            outputs = _clip_model.get_image_features(**inputs)
            embeds = _normalize(outputs).detach().cpu().numpy().astype(np.float32)
            out.append(embeds)
    return np.concatenate(out, axis=0) if out else np.zeros((0, _CLIP_DIM), dtype=np.float32)


def embed_queries(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Возвращает L2-нормированные эмбеддинги текстовых запросов в пространстве CLIP.
    """
    out: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            inputs = _processor(text=batch, padding=True, return_tensors="pt").to(_device)
            outputs = _clip_model.get_text_features(**inputs)
            embeds = _normalize(outputs).detach().cpu().numpy().astype(np.float32)
            out.append(embeds)
    return np.concatenate(out, axis=0) if out else np.zeros((0, _CLIP_DIM), dtype=np.float32)


def clip_dim() -> int:
    """
    Размерность CLIP эмбеддинга.
    """
    v = embed_queries(["_probe_"])
    return int(v.shape[1] if v.size else _CLIP_DIM)
