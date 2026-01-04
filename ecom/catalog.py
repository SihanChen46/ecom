"""Catalog management and file selection."""

import json
from pathlib import Path
from dataclasses import dataclass

from .client import GeminiClient


@dataclass
class ProductAssets:
    main_image: Path
    documents: list[Path]


class Catalog:
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    DOC_EXTENSIONS = {".pdf", ".txt", ".docx", ".doc"}
    IMAGE_FOLDERS = {"主图", "图片", "images", "photos"}

    def __init__(self, client: GeminiClient, base_dir: str = "catalog"):
        self.client = client
        self.base_dir = Path(base_dir)

    def get_product_dir(self, product_id: str) -> Path:
        path = self.base_dir / product_id
        if not path.exists():
            raise FileNotFoundError(f"Product not found: {product_id}")
        return path

    def get_assets(self, product_id: str) -> ProductAssets:
        """Get main image and relevant documents for a product."""
        product_dir = self.get_product_dir(product_id)
        return ProductAssets(
            main_image=self._select_main_image(product_dir),
            documents=self._select_documents(product_dir),
        )

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

    def _list_documents(self, product_dir: Path) -> list[Path]:
        docs = []
        for f in product_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in self.DOC_EXTENSIONS:
                docs.append(f)
        return sorted(docs)

    def _select_main_image(self, product_dir: Path) -> Path:
        images = self._list_images(self._find_image_folder(product_dir))
        if not images:
            raise FileNotFoundError(f"No images in: {product_dir}")
        if len(images) == 1:
            return images[0]

        prompt = f"""Select the main product image from: {json.dumps([i.name for i in images], ensure_ascii=False)}

Rules: Prefer 主图/main/正面图1. Avoid 细节/detail/尺寸/场景.
Reply with filename only."""

        selected = self.client.generate_text([prompt]).strip().strip("\"'")
        for img in images:
            if img.name == selected or selected in img.name:
                return img
        return images[0]

    def _select_documents(self, product_dir: Path) -> list[Path]:
        docs = self._list_documents(product_dir)
        if not docs:
            return []
        if len(docs) == 1:
            return docs

        prompt = f"""Select the most useful documents for understanding this product from: {json.dumps([d.name for d in docs], ensure_ascii=False)}

Prioritize:
1. 新建品基础信息 (product info)
2. 说明书 (manual)
3. Avoid: FCC, compliance docs

Reply with comma-separated filenames of the top 1-3 most useful docs."""

        selected = self.client.generate_text([prompt]).strip()
        selected_names = [s.strip().strip("\"'") for s in selected.split(",")]

        result = []
        for name in selected_names:
            for doc in docs:
                if doc.name == name or name in doc.name:
                    result.append(doc)
                    break
        return result if result else docs[:2]

    # Legacy method for backward compatibility
    def select_main_image(self, product_id: str) -> Path:
        return self.get_assets(product_id).main_image
