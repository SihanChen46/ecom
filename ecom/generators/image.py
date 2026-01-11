"""Image generation from prompts."""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..client import GeminiClient
from ..utils import load_image


# Different instructions for different modes
MODE_INSTRUCTIONS = {
    "cover": """Generate an image based on this prompt. Use the reference image as the product reference, maintain the product category but apply the creative style described.

{prompt}

Important: Keep the product recognizable as the same category but apply the creative transformation described above. The reference image shows the actual product.""",

    "preview": """Generate a realistic product photo based on this prompt. The product MUST look EXACTLY like the reference image - same shape, same details, same accessories, same colors.

{prompt}

CRITICAL: The product must be identical to the reference image. Only the lighting, angle, and environment can change. Do NOT modify the product itself in any way.""",

    "top": """Generate an e-commerce product image based on this natural language prompt. Use the reference image as the product reference.

{prompt}

IMPORTANT RULES:
1. Keep the product recognizable - same category and key features as the reference image.
2. For SIZE/DIMENSION shots: Maintain TRUE TO LIFE proportions. Do NOT exaggerate size. Render dimension text labels clearly and accurately.
3. Apply the creative styling, lighting, and composition described in the prompt.
4. This is for e-commerce - images should be professional, high-quality, and conversion-focused.
5. Use SQUARE 1:1 aspect ratio. Center the subject with breathing room on all sides.
6. If dimension labels are requested, render the text clearly and legibly.""",
}


class ImageGenerator:
    def __init__(self, client: GeminiClient, max_workers: int = 10):
        self.client = client
        self.max_workers = max_workers

    def generate(
        self,
        image_paths: list[str] | str,
        prompts: list[dict],
        output_dir: Path,
        mode: str = "cover",
    ) -> list[dict]:
        """
        Generate images from prompts using Nano Banana (parallel execution).
        
        Args:
            image_paths: Path(s) to reference product image(s). 
                        For 'top' mode, can be a list of paths (all sent to API).
                        For other modes, only first image is used.
            prompts: List of prompt dicts with 'name' and 'prompt' keys
            output_dir: Directory to save generated images
            mode: Generation mode - "cover", "preview", or "top"
            
        Returns:
            List of result dicts with generated image paths
        """
        # Normalize to list
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        
        # Create image parts for ALL images (for Hero Shot in 'top' mode)
        all_image_parts = []
        for img_path in image_paths:
            image_bytes, mime_type = load_image(img_path)
            all_image_parts.append(self.client.create_image_part(image_bytes, mime_type))
        
        # First image part only (for non-Hero shots and other modes)
        first_image_part = all_image_parts[:1]
        
        if mode == "top" and len(all_image_parts) > 1:
            print(f"      📷 Hero Shot will use {len(all_image_parts)} reference images")
            print("      📷 Other shots will use 1 reference image")
        
        total = len(prompts)
        results = [None] * total  # Pre-allocate to maintain order
        
        print(f"      🚀 Running {total} generations in parallel (max {self.max_workers} workers)...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_idx = {}
            for idx, item in enumerate(prompts):
                # For 'top' mode: Hero Shot (idx=0) gets all images, others get first only
                if mode == "top" and idx == 0:
                    image_parts = all_image_parts
                else:
                    image_parts = first_image_part
                
                future = executor.submit(
                    self._generate_one, idx, item, image_parts, output_dir, mode
                )
                future_to_idx[future] = idx
            
            # Collect results as they complete
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results[idx] = result
                    status = "✅" if result["images"] else "⚠️"
                    print(f"      [{idx+1}/{total}] {status} {result['name'][:50]}")
                except Exception as e:
                    results[idx] = {
                        "index": idx + 1,
                        "name": prompts[idx]["name"],
                        "images": [],
                        "error": str(e)
                    }
                    print(f"      [{idx+1}/{total}] ❌ Error: {e}")
            
        return results

    def _generate_one(
        self,
        idx: int,
        item: dict,
        image_parts: list,
        output_dir: Path,
        mode: str = "cover",
    ) -> dict:
        """Generate images for a single prompt."""
        name, prompt = item["name"], item["prompt"]
        result = {"index": idx + 1, "name": name, "images": [], "error": None}

        # Get the appropriate instruction template for the mode
        instruction_template = MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS["cover"])
        full_prompt = instruction_template.format(prompt=prompt)

        try:
            # Send all image parts + prompt
            img_result = self.client.generate_image([*image_parts, full_prompt])
            
            if not img_result.images:
                result["error"] = "No images generated"
                return result
                
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
                
        except Exception as e:
            result["error"] = str(e)

        return result
