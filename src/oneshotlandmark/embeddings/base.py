from abc import ABC, abstractmethod
from tqdm import tqdm
import torch
import logging
import time

logger = logging.getLogger(__name__)

class BaseEmbeddingGenerator(ABC):
    """
    Abstract base class for generating image embeddings at different
    granularities (patch-level, pixel-level, etc.).

    Each subclass implements `generate_embedding` which produces a flat
    (K, D) embedding tensor and a mapping from pixel coordinates to
    tensor indices.

    Args:
        verbose (bool): If True, show progress bars during batch processing.
    """

    def __init__(self, model, verbose=True):
        self.model=model
        self.verbose = verbose

    @abstractmethod
    def generate_embedding(self, img_path: str) -> tuple[torch.Tensor, dict]:
        """
        Generate embeddings for a single image.

        Args:
            img_path: Path to the image file.

        Returns:
            embeddings: Tensor of shape (K, D) where K is the number of
                candidate locations and D is the embedding dimension.
            xy_to_index: Dict mapping (x, y) pixel coordinates to indices
                in the embeddings tensor.
        """
        pass

    def generate_embedding_all(self, img_paths: list[str]) -> tuple[list[torch.Tensor], list[dict]]:
        """
        Generate embeddings for a list of images.

        Args:
            img_paths: List of image file paths.

        Returns:
            embeddings: List of (K_i, D) tensors, one per image.
            xy_to_index_maps: List of dicts mapping (x, y) -> index.
        """
        logger.info(f"Generating embeddings for {len(img_paths)} images")
        start = time.perf_counter()

        embeddings = []
        xy_maps = []
        iterator = (
            tqdm(img_paths, desc="Generating embeddings")
            if self.verbose
            else img_paths
        )
        for img_path in iterator:
            emb, xy_map = self.generate_embedding(img_path)
            embeddings.append(emb)
            xy_maps.append(xy_map)
            
        elapsed = time.perf_counter() - start
        total_K = sum(emb.shape[0] for emb in embeddings)
        logger.info(
            f"Generated embeddings for {len(img_paths)} images: "
            f"total_candidates={total_K}, time={elapsed:.2f}s "
            f"({elapsed / len(img_paths):.2f}s/image)"
        )

        return embeddings, xy_maps
