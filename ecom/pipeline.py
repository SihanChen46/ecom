"""Main pipeline orchestration."""

import json
from pathlib import Path

from .config import Config, MODES
from .client import GeminiClient
from .catalog import Catalog
from .generators import PromptGenerator, ImageGenerator, AllPromptGenerator, AdaptGenerator
from .utils import generate_task_id, load_image


class Pipeline:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.client = GeminiClient(self.config)
        self.catalog = Catalog(self.config.catalog_dir)
        self.prompt_gen = PromptGenerator(self.client)
        self.all_prompt_gen = AllPromptGenerator(self.client)
        self.image_gen = ImageGenerator(self.client)
        self.adapt_gen = AdaptGenerator(self.client)

    def run(self, file_paths: list[str], mode: str = "cover") -> dict:
        """
        Run the pipeline with a list of file paths.
        
        Args:
            file_paths: List of file paths (images or documents) from a product folder.
                       All files must be from the same product folder under catalog/.
            mode: Generation mode - "cover" (主图) or "preview" (预览图)
        
        Returns:
            dict with results including generated images.
        """
        if not file_paths:
            raise ValueError("At least one file path is required")
        
        if mode not in MODES:
            raise ValueError(f"Unknown mode: {mode}. Available: {list(MODES.keys())}")

        mode_names = {
            "cover": "主图 (Cover)",
            "preview": "预览图 (Preview)",
            "top": "全套11张 (Top 11-Shot Deck)",
            "adapt": "色彩适配 (Color Adapt)",
        }
        mode_name = mode_names.get(mode, mode)
        print(f"🎨 Mode: {mode_name}")
        
        # Adapt mode has a completely different flow
        if mode == "adapt":
            return self._run_adapt(file_paths)

        # Infer product ID from file paths
        product_id = self.catalog.infer_product_id(file_paths)
        print(f"📦 Inferred product ID: {product_id}")

        # Separate images and documents
        images, documents = self.catalog.classify_files(file_paths)
        
        if not images:
            raise ValueError("At least one image is required")
        
        # For 'top' mode, use all images; for other modes, use only first
        if mode == "top":
            print(f"🖼️  Images ({len(images)}):")
            for img in images:
                print(f"      - {img}")
        else:
            print(f"🖼️  Main image: {images[0]}")
        print(f"📄 Documents: {documents}")

        # Setup output directory - include mode in path
        product_dir = Path(self.config.output_dir) / product_id / mode
        product_dir.mkdir(parents=True, exist_ok=True)

        task_id = generate_task_id()
        task_dir = product_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        # Copy reference image(s)
        main_image = images[0]
        if mode == "top":
            # Save all reference images for 'top' mode
            for i, img_path in enumerate(images):
                image_bytes, _ = load_image(img_path)
                ref_path = task_dir / f"reference_{i+1:02d}{Path(img_path).suffix}"
                ref_path.write_bytes(image_bytes)
        else:
            # Save only main image for other modes
            image_bytes, _ = load_image(main_image)
            ref_path = task_dir / f"reference{Path(main_image).suffix}"
            ref_path.write_bytes(image_bytes)

        # Step 1: Generate image prompts
        print("\n🔄 Step 1: Generating image prompts...")
        
        # Prompts are saved directly in task_dir
        prompts_file = task_dir / "prompts.json"
        
        # Get the prompt file for the selected mode
        meta_prompt_path = self.config.get_prompt_file(mode)
        
        # Use AllPromptGenerator for 'top' mode (with all images), PromptGenerator for others
        if mode == "top":
            raw_analysis, prompts = self.all_prompt_gen.generate(
                images,  # Pass ALL images
                meta_prompt_path, 
                documents
            )
        else:
            raw_analysis, prompts = self.prompt_gen.generate(
                main_image,  # Pass only first image
                meta_prompt_path, 
                documents
            )
        # Save prompts in task_dir
        prompts_file.write_text(json.dumps(prompts, indent=2, ensure_ascii=False))
        if raw_analysis:
            (task_dir / "analysis.txt").write_text(raw_analysis)
        print(f"   ✅ Generated {len(prompts)} prompts")

        # Step 2: Generate images from prompts
        print("\n🔄 Step 2: Generating images...")
        if prompts:
            # For 'top' mode, pass all images; for others, pass first image
            ref_images = images if mode == "top" else main_image
            results = self.image_gen.generate(ref_images, prompts, task_dir, mode)
            images_generated = sum(len(r["images"]) for r in results)
            print(f"   ✅ Generated {images_generated} images")
        else:
            results = []
            print("   ⚠️  No prompts to generate images from")

        # Save results
        output = {
            "product_id": product_id,
            "task_id": task_id,
            "mode": mode,
            "reference_images": images if mode == "top" else [main_image],
            "documents": documents,
            "prompts_count": len(prompts),
            "images_generated": sum(len(r["images"]) for r in results),
            "results": results,
        }
        (task_dir / "results.json").write_text(json.dumps(output, indent=2, ensure_ascii=False))

        print("\n✅ Pipeline complete!")
        print(f"   Output directory: {task_dir}")

        return output

    def _infer_product_id_flexible(self, file_path: str) -> str:
        """
        Infer product ID from file path, supporting both catalog and outputs folders.
        
        Supports:
        - catalog/{product_id}/...
        - outputs/{product_id}/...
        """
        path = Path(file_path)
        parts = path.parts
        
        # Look for 'catalog' or 'outputs' in the path
        for folder_name in ['catalog', 'outputs']:
            try:
                idx = parts.index(folder_name)
                # Product ID is the next part after the folder
                if idx + 1 < len(parts):
                    return parts[idx + 1]
            except ValueError:
                continue
        
        # Fallback: use parent folder name
        return path.parent.name

    def _run_adapt(self, file_paths: list[str]) -> dict:
        """
        Run adapt mode: combine target image composition with product image colors.
        
        Args:
            file_paths: First image is target (composition reference), 
                       rest are product images (color/form reference)
        
        Returns:
            dict with results including generated images.
        """
        # Separate images (adapt mode doesn't use documents)
        images, _ = self.catalog.classify_files(file_paths)
        
        if len(images) < 2:
            raise ValueError("Adapt mode requires at least 2 images: 1 target + N product images")
        
        target_image = images[0]
        product_images = images[1:]
        
        print(f"🎯 Target image (composition): {target_image}")
        print(f"📦 Product images ({len(product_images)}):")
        for img in product_images:
            print(f"      - {img}")
        
        # Infer product ID from catalog or outputs folder
        product_id = self._infer_product_id_flexible(product_images[0])
        print(f"📦 Product ID: {product_id}")
        
        # Setup output directory
        product_dir = Path(self.config.output_dir) / product_id / "adapt"
        product_dir.mkdir(parents=True, exist_ok=True)
        
        task_id = generate_task_id()
        task_dir = product_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        
        # Save target image
        target_bytes, _ = load_image(target_image)
        target_ref_path = task_dir / f"target{Path(target_image).suffix}"
        target_ref_path.write_bytes(target_bytes)
        
        # Save product images
        for i, img_path in enumerate(product_images):
            img_bytes, _ = load_image(img_path)
            ref_path = task_dir / f"product_{i+1:02d}{Path(img_path).suffix}"
            ref_path.write_bytes(img_bytes)
        
        # Generate adapted images (no prompt generation step)
        print("\n🔄 Generating adapted images...")
        results = self.adapt_gen.generate(target_image, product_images, task_dir)
        images_generated = sum(len(r["images"]) for r in results)
        print(f"   ✅ Generated {images_generated} images")
        
        # Save results
        output = {
            "product_id": product_id,
            "task_id": task_id,
            "mode": "adapt",
            "target_image": target_image,
            "product_images": product_images,
            "images_generated": images_generated,
            "results": results,
        }
        (task_dir / "results.json").write_text(json.dumps(output, indent=2, ensure_ascii=False))
        
        print("\n✅ Pipeline complete!")
        print(f"   Output directory: {task_dir}")
        
        return output
