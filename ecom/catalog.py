"""Catalog management and image selection."""

import json
from pathlib import Path

from .client import GeminiClient


class Catalog:
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    IMAGE_FOLDERS = {"主图", "图片", "images", "photos"}

    def __init__(self, client: GeminiClient, base_dir: str = "catalog"):
        self.client = client
        self.base_dir = Path(base_dir)

    def get_product_dir(self, product_id: str) -> Path:
        path = self.base_dir / product_id
        if not path.exists():
            raise FileNotFoundError(f"Product not found: {product_id}")
        return path

    def _find_image_folder(self, product_dir: Path) -> Path:
        for name in self.IMAGE_FOLDERS:
            folder = product_dir / name
            if folder.is_dir():
                return folder
        return product_dir

    def _list_images(self, folder: Path) -> list[Path]:
        return sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in self.IMAGE_EXTENSIONS
        )

    def select_main_image(self, product_id: str) -> Path:
        product_dir = self.get_product_dir(product_id)
        images = self._list_images(self._find_image_folder(product_dir))

        if not images:
            raise FileNotFoundError(f"No images found for: {product_id}")
        if len(images) == 1:
            return images[0]

        prompt = self._build_selection_prompt([img.name for img in images])
        selected = self.client.generate_text([prompt]).strip().strip("\"'")

        for img in images:
            if img.name == selected or selected in img.name:
                return img
        return images[0]

    def _build_selection_prompt(self, names: list[str]) -> str:
        return f"""Select the main product image from: {json.dumps(names, ensure_ascii=False)}

Rules:
1. Prefer: 主图, main, primary, 正面图1
2. Avoid: 细节, detail, 尺寸, size, 场景, back
3. If numbered, prefer first

Reply with filename only."""

