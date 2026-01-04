#!/usr/bin/env python3
"""Generate e-commerce marketing images using Gemini AI."""

import argparse
import sys

from ecom import Pipeline, Config
from ecom.config import MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-p", "--product", help="Product ID from catalog")
    group.add_argument("-i", "--image", help="Direct image path")
    
    parser.add_argument("-n", "--num", type=int, help="Number of images to generate")
    parser.add_argument("-m", "--meta-prompt", help="Custom meta prompt file")
    parser.add_argument("--model", choices=MODELS.keys(), default="gemini-3")
    parser.add_argument("--catalog", default="catalog")

    args = parser.parse_args()

    config = Config.with_model(args.model, catalog_dir=args.catalog)
    pipeline = Pipeline(config)
    
    result = pipeline.run(
        product_id=args.product,
        image_path=args.image,
        meta_prompt_path=args.meta_prompt,
        num_images=args.num,
    )

    print(f"\n[{result['product_id']}] Generated {result['images_generated']} images")
    success = sum(1 for r in result["results"] if r["images"])
    print(f"Success: {success}/{len(result['results'])}")


if __name__ == "__main__":
    main()
