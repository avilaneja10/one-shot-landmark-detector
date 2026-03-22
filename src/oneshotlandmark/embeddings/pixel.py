from oneshotlandmark.embeddings.base import BaseEmbeddingGenerator
import logging
import time
import torch
from PIL import Image
from oneshotlandmark.utils import load_image
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class PixelEmbeddingGenerator(BaseEmbeddingGenerator):
    """
    Generates one embedding per pixel by tiling shifted versions of the image.
 
    For a given patch_size P, this creates P*P padded variants of the input
    image, each shifted by a different (top, left) offset in [0, P-1]^2.
    Each variant is passed through the ViT model, and the patch-center
    embeddings are mapped back to original pixel coordinates.
 
    This ensures every pixel in the original image appears at the center of
    a patch in exactly one variant, giving it a dedicated embedding.
 
    Args:
        model: A ViTModel instance used to extract hidden states.
        patch_size (int): Size of each square patch in pixels.
        normalize (bool): If True, mean-center patch tokens before L2 normalization.
        batch_size (int): Number of padded variants to process per model forward pass.
        verbose (bool): If True, show progress bars during batch processing.
 
    Note:
        For patch_size=16, this generates 256 padded variants per image,
        making it ~256x more expensive than patch-level embedding generation.
    """
    def __init__(self, model, patch_size=16, normalize=True, batch_size=32, verbose=True):
        super().__init__(model, verbose)
        self.patch_size = patch_size
        self.normalize = normalize
        self.batch_size = batch_size

    def _generate_padded_images(self, img: Image.Image) -> list[Image.Image]:
        """
        Create patch_size^2 shifted variants of the input image.
 
        For each (top, left) offset in [0, patch_size-1]^2:
          - Add black padding of `top` pixels above and `left` pixels to the left
          - Add black padding on bottom/right so dimensions are multiples of patch_size
 
        This ensures every pixel in the original image appears at the center of
        a patch in exactly one variant.
 
        Args:
            img: RGB PIL image.
 
        Returns:
            List of patch_size^2 padded PIL images.
        """
        w0, h0 = img.size
        ps = self.patch_size
        padded_images = []
 
        for top in range(ps):
            for left in range(ps):
                pad_bottom = (ps - ((h0 + top) % ps)) % ps
                pad_right = (ps - ((w0 + left) % ps)) % ps
 
                new_w = w0 + left + pad_right
                new_h = h0 + top + pad_bottom
 
                padded = Image.new("RGB", (new_w, new_h), (0, 0, 0))
                padded.paste(img, (left, top))
                padded_images.append(padded)
 
        return padded_images
    
    def generate_embedding(self, img_path: str) -> torch.Tensor:
        """
        Generate pixel-level embeddings for a single image.
 
        Steps:
            1. Load image and create P*P shifted/padded variants.
            2. Batch all variants through the ViT model.
            3. For each variant, extract and normalize patch tokens.
            4. Map each patch center back to its original pixel coordinate,
               writing directly into a pre-allocated (K, D) tensor.
 
        Args:
            img_path: Path to the image file.
 
        Returns:
            embeddings: (K, D) tensor of normalized pixel embeddings where
                K = orig_width * orig_height, indexed in sorted (x, y) order.
        """
        start = time.perf_counter()
        pil_image = load_image(img_path)
        orig_w, orig_h = pil_image.size

        # Step 1: Generate all shifted/padded variants
        padded_images = self._generate_padded_images(pil_image)
        n_variants = len(padded_images)

        logger.debug(
            f"Image '{img_path}': original={orig_w}x{orig_h}, "
            f"generated {n_variants} padded variants"
        )

        # Step 2: Batch forward pass through model
        t_model = time.perf_counter()
        all_hidden = self.model.generate_embedding_batch(
            padded_images, batch_size=self.batch_size
        )
        logger.debug(f"Model forward pass: {time.perf_counter() - t_model:.3f}s")

        # Step 3: Extract and normalize patch tokens for each variant
        t_norm = time.perf_counter()
        grid_embeddings = []
        for i, hidden in enumerate(all_hidden):
            padded_w, padded_h = padded_images[i].size
            grid_cols = padded_w // self.patch_size
            grid_rows = padded_h // self.patch_size
            K = grid_rows * grid_cols
 
            patch_tokens = hidden[-K:, :]  # (K, D)
 
            if self.normalize:
                patch_tokens = patch_tokens - patch_tokens.mean(dim=0, keepdim=True)
            patch_tokens = F.normalize(patch_tokens, p=2, dim=1, eps=1e-8)
 
            grid_embeddings.append(
                patch_tokens.reshape(grid_rows, grid_cols, -1)
            )
        logger.debug(f"Normalization: {time.perf_counter() - t_norm:.3f}s")

        # Step 4: Map patch centers to original pixel coordinates,
        #         writing directly into a pre-allocated tensor.
        #
        # Sorted (x, y) order is lexicographic: (0,0),(0,1),...,(1,0),...
        # so flat index = x * orig_h + y. Pre-allocating avoids building
        # an intermediate dict, sorting it, and calling torch.stack on a
        # list of ~300K tensors.
        t_map = time.perf_counter()
 
        D = grid_embeddings[0].shape[2]
        K = orig_w * orig_h
        embeddings = torch.zeros(K, D)
 
        ps = self.patch_size
        center_offset = ps // 2
 
        for idx, grid_emb in enumerate(grid_embeddings):
            top = idx // ps
            left = idx % ps
            grid_rows, grid_cols, _ = grid_emb.shape
 
            for row in range(grid_rows):
                orig_y = center_offset + row * ps - top
                if orig_y < 0 or orig_y >= orig_h:
                    continue
 
                for col in range(grid_cols):
                    orig_x = center_offset + col * ps - left
                    if orig_x < 0 or orig_x >= orig_w:
                        continue
 
                    flat_idx = orig_x * orig_h + orig_y
                    embeddings[flat_idx] = grid_emb[row, col]
 
        logger.debug(
            f"Pixel mapping: {embeddings.shape[0]} pixels, "
            f"{time.perf_counter() - t_map:.3f}s"
        )
 
        elapsed = time.perf_counter() - start
        logger.debug(
            f"Pixel embedding for '{img_path}': K={embeddings.shape[0]}, "
            f"D={embeddings.shape[1]}, time={elapsed:.3f}s"
        )
 
        del padded_images, all_hidden, grid_embeddings
        return embeddings