"""Main pipeline orchestration."""

import json
from pathlib import Path

from .config import Config
from .client import GeminiClient
from .catalog import Catalog
from .generators import PromptGenerator, ImageGenerator
from .utils import generate_task_id, load_image


class Pipeline:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.client = GeminiClient(self.config)
        self.catalog = Catalog(self.client, self.config.catalog_dir)
        self.prompt_gen = PromptGenerator(self.client)
        self.image_gen = ImageGenerator(self.client)

    def run(
        self,
        product_id: str = None,
        image_path: str = None,
        meta_prompt_path: str = None,
        num_images: int = None,
    ) -> dict:
        # Resolve image
        if image_path:
            main_image = image_path
            pid = Path(image_path).stem
        elif product_id:
            main_image = str(self.catalog.select_main_image(product_id))
            pid = product_id
        else:
            raise ValueError("product_id or image_path required")

        # Setup paths
        product_dir = Path(self.config.output_dir) / pid
        product_dir.mkdir(parents=True, exist_ok=True)
        
        task_id = generate_task_id()
        task_dir = product_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        # Copy reference
        image_bytes, _ = load_image(main_image)
        ref_path = task_dir / f"reference{Path(main_image).suffix}"
        ref_path.write_bytes(image_bytes)

        # Get prompts (reuse or generate)
        prompts_file = product_dir / "prompts.json"
        if prompts_file.exists():
            prompts = json.loads(prompts_file.read_text())
            raw_analysis = None
        else:
            meta_prompt_path = meta_prompt_path or self.config.meta_prompt_file
            raw_analysis, prompts = self.prompt_gen.generate(main_image, meta_prompt_path)
            prompts_file.write_text(json.dumps(prompts, indent=2, ensure_ascii=False))
            if raw_analysis:
                (task_dir / "analysis.txt").write_text(raw_analysis)

        # Limit prompts
        if num_images and num_images > 0:
            prompts = prompts[:num_images]

        # Generate images
        results = self.image_gen.generate(main_image, prompts, task_dir) if prompts else []

        # Save results
        output = {
            "product_id": pid,
            "task_id": task_id,
            "main_image": main_image,
            "prompts_used": len(prompts),
            "images_generated": sum(len(r["images"]) for r in results),
            "results": results,
        }
        (task_dir / "results.json").write_text(json.dumps(output, indent=2, ensure_ascii=False))

        return output

