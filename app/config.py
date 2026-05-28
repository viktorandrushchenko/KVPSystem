from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("KVP_DATA_DIR", ROOT_DIR / "data"))
UPLOAD_DIR = Path(os.getenv("KVP_UPLOAD_DIR", DATA_DIR / "uploads"))
PAGE_DIR = Path(os.getenv("KVP_PAGE_DIR", DATA_DIR / "pages"))
DB_PATH = Path(os.getenv("KVP_DB_PATH", DATA_DIR / "documents.sqlite3"))
CHECKPOINT_NAMES_PATH = Path(os.getenv("KVP_CHECKPOINT_NAMES_PATH", DATA_DIR / "checkpoint_names.json"))
ANNOTATION_EXPORT_DIR = Path(os.getenv("KVP_ANNOTATION_EXPORT_DIR", DATA_DIR / "annotation_exports"))
TRAINED_CHECKPOINT_DIR = Path(os.getenv("KVP_TRAINED_CHECKPOINT_DIR", ROOT_DIR / "trained_checkpoints"))

BASE_MODEL_PATH = Path(os.getenv("DEEPSEEK_OCR_BASE_MODEL", ROOT_DIR / "deepseek_ocr"))
ADAPTER_PATH = Path(os.getenv("DEEPSEEK_OCR_ADAPTER", ROOT_DIR / "checkpoint-1"))
_device_map = os.getenv("DEEPSEEK_OCR_DEVICE_MAP", "none").strip()
DEVICE_MAP = "" if _device_map.lower() in {"", "none", "false", "off"} else _device_map
MAX_MEMORY = os.getenv("DEEPSEEK_OCR_MAX_MEMORY", "0:5GiB,cpu:24GiB")
BASE_SIZE = int(os.getenv("DEEPSEEK_OCR_BASE_SIZE", "640"))
IMAGE_SIZE = int(os.getenv("DEEPSEEK_OCR_IMAGE_SIZE", "640"))
CROP_MODE = os.getenv("DEEPSEEK_OCR_CROP_MODE", "false").lower() in {"1", "true", "yes"}
MAX_NEW_TOKENS = int(os.getenv("DEEPSEEK_OCR_MAX_NEW_TOKENS", "128"))

DEFAULT_PROMPT = os.getenv(
    "KVP_EXTRACTION_PROMPT",
    (
        "<image>\n"
        "Extract all key-value pairs from this document. "
        "Return only valid JSON as an array of objects with keys: key, value, confidence."
    ),
)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".pdf"}
