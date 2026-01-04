"""Image generation from prompts."""

import re
from pathlib import Path

from ..client import GeminiClient, TokenUsage
from ..utils import load_image


class ImageGenerator:
    def __init__(self, client: GeminiClient):
        self.client = client

    def generate(
        self,
        image_path: str,
        prompts: list[dict],
        output_dir: Path,
    ) -> tuple[list[dict], TokenUsage]:
        image_bytes, mime_type = load_image(image_path)
        image_part = self.client.create_image_part(image_bytes, mime_type)
        image_tokens = self.client.estimate_image_tokens(image_bytes)
        
        results = []
        total_usage = TokenUsage()
        
        for idx, item in enumerate(prompts):
            result, usage = self._generate_one(idx, item, image_part, image_tokens, output_dir)
            results.append(result)
            total_usage = total_usage + usage
            
        return results, total_usage

    def _generate_one(
        self,
        idx: int,
        item: dict,
        image_part,
        ref_image_tokens: int,
        output_dir: Path,
    ) -> tuple[dict, TokenUsage]:
        name, prompt = item["name"], item["prompt"]
        result = {"index": idx + 1, "name": name, "images": [], "error": None}
        
        text_tokens = self.client.estimate_text_tokens(prompt)
        content_info = {
            "text_tokens": text_tokens,
            "image_tokens": ref_image_tokens,
            "document_tokens": 0,
        }

        try:
            img_result = self.client.generate_image([image_part, prompt], content_info=content_info)
            for img_data, mime in img_result.images:
                ext = {
                    "image/png": ".png",
                    "image/jpeg": ".jpg",
                    "image/webp": ".webp",
                }.get(mime, ".png")

                safe_name = re.sub(r"[^\w\-]", "_", name)[:30]
                filepath = output_dir / f"{idx + 1:02d}_{safe_name}{ext}"
                filepath.write_bytes(img_data)
                result["images"].append(str(filepath))
            # Set output stats
            img_result.usage.output_image_count = len(img_result.images)
            return result, img_result.usage
        except Exception as e:
            result["error"] = str(e)
            return result, TokenUsage()

