"""Image generation from prompts."""

import re
from pathlib import Path

from ..client import GeminiClient
from ..utils import load_image


class ImageGenerator:
    def __init__(self, client: GeminiClient):
        self.client = client

    def generate(
        self,
        image_path: str,
        prompts: list[dict],
        output_dir: Path,
    ) -> list[dict]:
        image_bytes, mime_type = load_image(image_path)
        image_part = self.client.create_image_part(image_bytes, mime_type)
        
        results = []
        for idx, item in enumerate(prompts):
            result = self._generate_one(idx, item, image_part, output_dir)
            results.append(result)
        return results

    def _generate_one(
        self,
        idx: int,
        item: dict,
        image_part,
        output_dir: Path,
    ) -> dict:
        name, prompt = item["name"], item["prompt"]
        result = {"index": idx + 1, "name": name, "images": [], "error": None}

        try:
            images = self.client.generate_image([image_part, prompt])
            for img_data, mime in images:
                ext = {
                    "image/png": ".png",
                    "image/jpeg": ".jpg",
                    "image/webp": ".webp",
                }.get(mime, ".png")

                safe_name = re.sub(r"[^\w\-]", "_", name)[:30]
                filepath = output_dir / f"{idx + 1:02d}_{safe_name}{ext}"
                filepath.write_bytes(img_data)
                result["images"].append(str(filepath))
        except Exception as e:
            result["error"] = str(e)

        return result

