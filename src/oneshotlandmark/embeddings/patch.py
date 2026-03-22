from oneshotlandmark.embeddings.base import BaseEmbeddingGenerator
import torch
from oneshotlandmark.utils import load_image, pad_to_multiple
import torch.nn.functional as F
import logging
import time

logger = logging.getLogger(__name__)

class PatchEmbeddingGenerator(BaseEmbeddingGenerator):
    """
    Generates one embedding per non-overlapping patch in an image.

    The image is padded so its dimensions are exact multiples of patch_size,
    then passed through the ViT model. The resulting patch tokens are optionally
    mean-centered and always L2-normalized.

    Args:
        model: A ViTModel instance used to extract hidden states.
        patch_size (int): Size of each square patch in pixels.
        normalize (bool): If True, mean-center patch tokens before L2 normalization.
        verbose (bool): If True, show progress bars during batch processing.
    """
    def __init__(self, model, patch_size=16, normalize=True, verbose=True):
        super().__init__(model, verbose)
        self.patch_size = patch_size
        self.normalize = normalize

    def generate_embedding(self, img_path: str) -> torch.Tensor:
        """
        Generate patch-level embeddings for a single image.

        Steps:
            1. Load image and pad to nearest multiple of patch_size.
            2. Extract hidden states from the ViT model.
            3. Isolate the last K patch tokens.
            4. Optionally mean-center, then L2-normalize. 
        Args:
            img_path: Path to the image file.

        Returns:
            embeddings: (K, D) tensor of normalized patch embeddings.
        TODO : Since it extends BaseEmbeddingGenerator, if we call it over all images
        it will do a forward pass one by one, optimal can be to batch them if of same sizes
        However, patch level is inherently fast, so keeping this as it is.
        """
        start = time.perf_counter()
        pil_image = load_image(img_path)
        orig_w, orig_h = pil_image.size
        padded_image = pad_to_multiple(pil_image, self.patch_size)
        padded_w, padded_h = padded_image.size
        grid_cols = padded_w // self.patch_size
        grid_rows = padded_h // self.patch_size
        K = grid_rows * grid_cols
        logger.debug(
            f"Image '{img_path}': original={orig_w}x{orig_h}, "
            f"padded={padded_w}x{padded_h}, grid={grid_rows}x{grid_cols}, K={K}"
        )
        # Extract hidden states: (num_tokens, D) where first token is CLS
        hidden = self.model.generate_embedding(padded_image)
        patch_tokens = hidden[-K:, :]  # (K, D)
        # Normalize: optional mean-centering followed by L2 normalization
        if self.normalize:
            patch_tokens = patch_tokens - patch_tokens.mean(dim=0, keepdim=True)

        patch_tokens = F.normalize(patch_tokens, p=2, dim=1, eps=1e-8)  # (K, D)
        elapsed = time.perf_counter() - start
        logger.debug(
            f"Patch embedding for '{img_path}': K={K}, D={patch_tokens.shape[1]}, "
            f"time={elapsed:.3f}s"
        )

        return patch_tokens