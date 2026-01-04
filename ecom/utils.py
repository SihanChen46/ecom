"""Utility functions."""

import mimetypes
from datetime import datetime
from pathlib import Path


def generate_task_id() -> str:
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def load_image(path: str) -> tuple[bytes, str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    mime, _ = mimetypes.guess_type(str(p))
    return p.read_bytes(), mime or "image/jpeg"


def load_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return p.read_text(encoding="utf-8")

