"""Catalog management and file classification."""

from pathlib import Path


class Catalog:
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    DOC_EXTENSIONS = {".pdf", ".txt", ".docx", ".doc"}

    def __init__(self, base_dir: str = "catalog"):
        self.base_dir = Path(base_dir)

    def infer_product_id(self, file_paths: list[str]) -> str:
        """
        Infer product ID from file paths.
        Assumes all files are under catalog/{product_id}/...
        """
        if not file_paths:
            raise ValueError("No file paths provided")

        # Get the first path and extract product ID
        first_path = Path(file_paths[0])
        
        # Find the catalog directory in the path
        parts = first_path.parts
        try:
            catalog_idx = parts.index(self.base_dir.name)
            # Product ID is the next part after "catalog"
            if catalog_idx + 1 < len(parts):
                return parts[catalog_idx + 1]
        except ValueError:
            pass
        
        # Fallback: use parent directory name
        # If path is like: catalog/PRODUCT_ID/subfolder/file.jpg
        # We need to find the product_id level
        for path_str in file_paths:
            p = Path(path_str)
            # Check if any parent matches a product folder
            for parent in p.parents:
                if parent.parent == self.base_dir or parent.parent.name == self.base_dir.name:
                    return parent.name
        
        raise ValueError(f"Cannot infer product ID from paths: {file_paths}")

    def classify_files(self, file_paths: list[str]) -> tuple[list[str], list[str]]:
        """
        Classify files into images and documents.
        
        Returns:
            (images, documents) tuple of lists
        """
        images = []
        documents = []
        
        for path_str in file_paths:
            path = Path(path_str)
            suffix = path.suffix.lower()
            
            if suffix in self.IMAGE_EXTENSIONS:
                images.append(path_str)
            elif suffix in self.DOC_EXTENSIONS:
                documents.append(path_str)
            else:
                # Unknown type - try to classify by content or skip
                print(f"⚠️  Unknown file type, skipping: {path_str}")
        
        return images, documents

    def get_product_dir(self, product_id: str) -> Path:
        """Get the product directory path."""
        path = self.base_dir / product_id
        if not path.exists():
            raise FileNotFoundError(f"Product not found: {product_id}")
        return path
