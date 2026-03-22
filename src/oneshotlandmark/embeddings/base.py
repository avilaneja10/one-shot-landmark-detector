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
    Each subclass implements generate_embedding which produces a flat
    (K, D) embedding tensor and a mapping from pixel coordinates to
    tensor indices.
    Args:
        verbose (bool): If True, show progress bars during batch processing.
    """
    def __init__(self, model, verbose=True):
        self.model=model
        self.verbose = verbose
    @abstractmethod
    def generate_embedding(self, img_path: str) -> torch.Tensor:
        """
        ...
        Returns:
            embeddings: Tensor of shape (K, D)
        """
        pass
    def generate_embedding_all(self, img_paths: list[str]) -> list[torch.Tensor]:
        """
        Generate embeddings for a list of images.
        Args:
            img_paths: List of image file paths.
        Returns:
            embeddings: List of (K_i, D) tensors.
        """
        logger.info(f"Generating embeddings for {len(img_paths)} images")
        start = time.perf_counter()
        embeddings = []
        iterator = (
            tqdm(img_paths, desc="Generating embeddings")
            if self.verbose
            else img_paths
        )

        for path in iterator:
            emb = self.generate_embedding(path)
            embeddings.append(emb)

        elapsed = time.perf_counter() - start
        total_K = sum(emb.shape[0] for emb in embeddings)
        logger.info(
            f"Generated embeddings for {len(img_paths)} images: "
            f"total_candidates={total_K}, time={elapsed:.2f}s "
            f"({elapsed / len(img_paths):.2f}s/image)"
        )
        return embeddings