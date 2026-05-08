"""
api/image_utils.py
------------------
Image loading, source detection (Mobicap vs smartphone), header cropping,
GPS extraction, and preprocessing for model inference.
"""

import sys
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.extract_gps import (
    _has_blue_header_img,
    _extract_exif_gps_img,
    _extract_ocr_gps_img,
)
from src.data.dataset import get_val_transforms

_TRANSFORM = get_val_transforms(image_size=224)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_image(data: bytes) -> Image.Image:
    """Load raw bytes into a PIL RGB image."""
    return Image.open(BytesIO(data)).convert("RGB")


# ---------------------------------------------------------------------------
# Source detection
# ---------------------------------------------------------------------------

def detect_source(img: Image.Image) -> str:
    """Return 'mobicap' if image has a blue Mobicap header, else 'smartphone'."""
    return "mobicap" if _has_blue_header_img(img) else "smartphone"


# ---------------------------------------------------------------------------
# GPS extraction
# ---------------------------------------------------------------------------

def extract_gps(img: Image.Image) -> tuple[float | None, float | None, str | None]:
    """
    Try EXIF GPS first (smartphone), then OCR header (Mobicap).
    Returns (lat, lon, source) where source is 'exif' | 'ocr' | None.
    """
    lat, lon = _extract_exif_gps_img(img)
    if lat is not None and lon is not None:
        return lat, lon, "exif"

    if _has_blue_header_img(img):
        lat, lon = _extract_ocr_gps_img(img)
        if lat is not None and lon is not None:
            return lat, lon, "ocr"

    return None, None, None


# ---------------------------------------------------------------------------
# Preprocessing for model input
# ---------------------------------------------------------------------------

def preprocess(img: Image.Image, is_mobicap: bool) -> torch.Tensor:
    """
    Crop the Mobicap blue header (top 1/12 of image) if applicable,
    apply ImageNet-normalised val transform, return (1, 3, 224, 224) tensor.
    """
    if is_mobicap:
        w, h = img.size
        img  = img.crop((0, h // 12, w, h))
    return _TRANSFORM(img).unsqueeze(0)   # (1, 3, H, W)
