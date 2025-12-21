"""
OCR для изображений с помощью pytesseract.
"""

from os import getenv
from pathlib import Path

from PIL import Image
import pytesseract


def ocr_image(path: Path) -> str:
    """
    Читает изображение и возвращает распознанный текст.
    """
    lang = (getenv("OCR_LANG") or "rus+eng").strip()
    img = Image.open(path).convert("RGB")
    text = pytesseract.image_to_string(img, lang=lang)
    return text.strip()
