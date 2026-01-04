"""Main pipeline orchestration."""

import json
from pathlib import Path

from .config import Config
from .client import GeminiClient, TokenUsage
from .catalog import Catalog
from .generators import PromptGenerator, ImageGenerator, TitleGenerator
from .utils import generate_task_id, load_image


class Pipeline:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.client = GeminiClient(self.config)
        self.catalog = Catalog(self.client, self.config.catalog_dir)
        self.prompt_gen = PromptGenerator(self.client)
        self.image_gen = ImageGenerator(self.client)
        self.title_gen = TitleGenerator(self.client)

    def run(
        self,
        product_id: str = None,
        image_path: str = None,
        meta_prompt_path: str = None,
        num_images: int = None,
    ) -> dict:
        # Token usage tracking
        usage_breakdown = {
            "prompt_generation": None,
            "title_generation": None, 
            "image_generation": None,
        }
        total_usage = TokenUsage()
        
        # Resolve assets
        if image_path:
            main_image = image_path
            documents = []
            pid = Path(image_path).stem
        elif product_id:
            assets = self.catalog.get_assets(product_id)
            main_image = str(assets.main_image)
            documents = assets.documents
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
            raw_analysis, prompts, prompt_usage = self.prompt_gen.generate(
                main_image, meta_prompt_path, documents
            )
            prompts_file.write_text(json.dumps(prompts, indent=2, ensure_ascii=False))
            if raw_analysis:
                (task_dir / "analysis.txt").write_text(raw_analysis)
            usage_breakdown["prompt_generation"] = prompt_usage.to_dict()
            total_usage = total_usage + prompt_usage

        # Limit prompts
        if num_images and num_images > 0:
            prompts = prompts[:num_images]

        # Generate images
        if prompts:
            results, image_usage = self.image_gen.generate(main_image, prompts, task_dir)
            usage_breakdown["image_generation"] = image_usage.to_dict()
            total_usage = total_usage + image_usage
        else:
            results = []

        # Generate SEO titles
        titles_file = product_dir / "titles.json"
        if titles_file.exists():
            titles = json.loads(titles_file.read_text())
        else:
            titles, title_usage = self.title_gen.generate(
                main_image, self.config.title_prompt_file, documents
            )
            titles_file.write_text(json.dumps(titles, indent=2, ensure_ascii=False))
            usage_breakdown["title_generation"] = title_usage.to_dict()
            total_usage = total_usage + title_usage

        # Calculate token source breakdown
        token_sources = self._calculate_token_sources(usage_breakdown)
        
        # Calculate costs with model info
        models_used = {
            "text_model": self.config.model_text,
            "image_model": self.config.model_image,
        }
        total_cost = total_usage.calculate_cost(self.config.model_image)

        # Save results
        output = {
            "product_id": pid,
            "task_id": task_id,
            "main_image": main_image,
            "documents": [str(d) for d in documents],
            "prompts_used": len(prompts),
            "images_generated": sum(len(r["images"]) for r in results),
            "titles": titles,
            "results": results,
            "token_usage": {
                "total": total_usage.to_dict(model=self.config.model_image),
                "by_stage": usage_breakdown,
                "by_source": token_sources,
                "models": models_used,
                "cost_usd": total_cost,
            },
        }
        (task_dir / "results.json").write_text(json.dumps(output, indent=2, ensure_ascii=False))
        
        # Print token usage summary
        self._print_token_summary(output["token_usage"], models_used)

        return output
    
    def _calculate_token_sources(self, breakdown: dict) -> dict:
        """Calculate token consumption by source type."""
        text_total = 0
        image_total = 0
        document_total = 0
        
        for stage, usage in breakdown.items():
            if usage and "input" in usage:
                text_total += usage["input"].get("text_tokens", 0)
                image_total += usage["input"].get("image_tokens", 0)
                document_total += usage["input"].get("document_tokens", 0)
        
        total = text_total + image_total + document_total
        
        return {
            "prompt_text": {
                "tokens": text_total,
                "percentage": f"{(text_total / total * 100):.1f}%" if total > 0 else "0%",
            },
            "images": {
                "tokens": image_total,
                "percentage": f"{(image_total / total * 100):.1f}%" if total > 0 else "0%",
            },
            "documents": {
                "tokens": document_total,
                "percentage": f"{(document_total / total * 100):.1f}%" if total > 0 else "0%",
            },
        }
    
    def _print_token_summary(self, usage: dict, models: dict = None) -> None:
        """Print a formatted token usage summary."""
        print("\n" + "=" * 70)
        print("📊 TOKEN USAGE & COST SUMMARY")
        print("=" * 70)
        
        # Models info
        if models:
            print(f"\n🤖 Models: text={models.get('text_model', 'N/A')}, image={models.get('image_model', 'N/A')}")
        
        total = usage["total"]
        print(f"\n📈 Total: {total['total_tokens']:,} tokens")
        print(f"   ├─ Input: {total['prompt_tokens']:,}")
        print(f"   └─ Output: {total['completion_tokens']:,}")
        
        # Detailed per-stage breakdown
        stage_names = {
            "prompt_generation": "Stage 1: Prompt Generation",
            "title_generation": "Stage 2: Title Generation",
            "image_generation": "Stage 3: Image Generation",
        }
        
        print("\n" + "-" * 70)
        for stage_key, stage_name in stage_names.items():
            data = usage["by_stage"].get(stage_key)
            if data:
                print(f"\n🔹 {stage_name}")
                print(f"   Total: {data['total_tokens']:,} tokens")
                
                # Input breakdown
                inp = data.get("input", {})
                print("   📥 Input:")
                print(f"      ├─ Text (prompt):    {inp.get('text_tokens', 0):,}")
                print(f"      ├─ Image:            {inp.get('image_tokens', 0):,}")
                print(f"      └─ Document:         {inp.get('document_tokens', 0):,}")
                
                # Output breakdown
                out = data.get("output", {})
                completion = data.get('completion_tokens', 0)
                print("   📤 Output:")
                if stage_key == "image_generation":
                    print(f"      ├─ Tokens:           {completion:,}")
                    print(f"      └─ Images generated: {out.get('image_count', 0)}")
                else:
                    print(f"      └─ Text tokens:      {completion:,}")
        
        # Overall source summary
        print("\n" + "-" * 70)
        print("\n📦 Input Token Summary (All Stages):")
        sources = usage["by_source"]
        print(f"   ├─ Text (prompt): {sources['prompt_text']['tokens']:,} ({sources['prompt_text']['percentage']})")
        print(f"   ├─ Images:        {sources['images']['tokens']:,} ({sources['images']['percentage']})")
        print(f"   └─ Documents:     {sources['documents']['tokens']:,} ({sources['documents']['percentage']})")
        
        # Cost breakdown
        cost = usage.get("cost_usd", {})
        if cost:
            print("\n" + "-" * 70)
            print("\n💰 COST BREAKDOWN (USD):")
            print("   📎 Pricing: https://ai.google.dev/pricing")
            inp_cost = cost.get("input", {})
            out_cost = cost.get("output", {})
            
            print("\n   📥 Input:")
            print(f"      ├─ Text:      ${inp_cost.get('text', 0):.6f}")
            print(f"      ├─ Image:     ${inp_cost.get('image', 0):.6f}")
            print(f"      ├─ Document:  ${inp_cost.get('document', 0):.6f}")
            print(f"      └─ Subtotal:  ${inp_cost.get('subtotal', 0):.6f}")
            
            print("   📤 Output (text + generated images):")
            print(f"      ├─ Tokens:    ${out_cost.get('text_and_image', 0):.6f}")
            if out_cost.get('imagen_per_image', 0) > 0:
                print(f"      ├─ Imagen:    ${out_cost.get('imagen_per_image', 0):.6f}")
            print(f"      └─ Subtotal:  ${out_cost.get('subtotal', 0):.6f}")
            
            total_cost = cost.get("total", 0)
            print(f"\n   💵 TOTAL COST: ${total_cost:.4f}")
            
            # Show note
            if total_cost == 0:
                print("   ℹ️  (Using free/experimental tier)")
            else:
                print("   ℹ️  Note: Generated images are charged via output tokens (~1,290+ tokens/image)")
        
        print("=" * 70 + "\n")
