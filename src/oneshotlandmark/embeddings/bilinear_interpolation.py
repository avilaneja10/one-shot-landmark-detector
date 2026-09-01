from oneshotlandmark.embeddings.base import BaseEmbeddingGenerator
from oneshotlandmark.embeddings.xy_map import LazyXYMapping
import torch
import torch.nn.functional as F
from oneshotlandmark.utils import load_image, pad_to_multiple
import logging
import time

logger = logging.getLogger(__name__)


class BilinearEmbeddingGenerator(BaseEmbeddingGenerator):
    """
    Generates pixel-level embeddings via a single ViT forward pass followed
    by bilinear interpolation of the patch-token grid.

    PixelEmbeddingGenerator achieves pixel-level resolution by running P²
    shifted forward passes (one per (top, left) offset in [0, P-1]²) so that
    every pixel appears at the centre of exactly one patch.  This class
    achieves the same spatial resolution with a SINGLE forward pass: the coarse
    (grid_rows × grid_cols, D) patch grid is upsampled to pixel resolution
    using bilinear interpolation.

    Alignment detail
    ----------------
    The image is padded to a multiple of patch_size before the forward pass, so
    the patch grid covers the full padded extent (padded_h × padded_w).
    Upsampling is therefore done to (padded_h, padded_w) rather than directly
    to (orig_h, orig_w); the result is then cropped back to the original
    dimensions.  With align_corners=False this ensures pixel p maps to patch
    coordinate (p + 0.5) / patch_size - 0.5, so patch centres land exactly at
    their integer grid positions.

    Output convention
    -----------------
    The returned (K, D) tensor and xy_to_index dict use the same column-major
    layout as PixelEmbeddingGenerator:

        flat_idx = x * orig_h + y      (x — column, y — row)

    This means the two generators are drop-in replacements for every downstream
    component (ScoreGenerator, CP methods, etc.) with no code changes required.

    Cost comparison
    ---------------
    PixelEmbeddingGenerator : P² forward passes  (256 for P=16)
    BilinearEmbeddingGenerator : 1 forward pass  + cheap CPU interpolation

    Args:
        model: A ViTModel instance used to extract hidden states.
        patch_size (int): Size of each square ViT patch in pixels.
        normalize (bool): If True, mean-centre embeddings before L2 normalisation.
        verbose (bool): If True, show progress bars during batch processing.
    """

    def __init__(self, model, patch_size: int = 16, normalize: bool = True, verbose: bool = True):
        super().__init__(model, verbose)
        self.patch_size = patch_size
        self.normalize = normalize

    def generate_embedding(self, img_path: str) -> tuple[torch.Tensor, dict]:
        """
        Generate pixel-level embeddings for a single image.

        Steps:
            1. Load image; pad to nearest multiple of patch_size.
            2. Single ViT forward pass → (K_patch, D) patch tokens.
            3. Reshape to (1, D, grid_rows, grid_cols) for F.interpolate.
            4. Bilinear upsample to (1, D, padded_h, padded_w).
               align_corners=False ensures patch centre p maps to grid index
               (p + 0.5) / patch_size - 0.5, i.e. integer grid positions.
            5. Crop to (1, D, orig_h, orig_w).
            6. Permute + reshape to (K, D) with K = orig_w * orig_h, using
               column-major order: flat_idx = x * orig_h + y.
            7. Optional mean-centring followed by L2 normalisation.
            8. Build xy_to_index dict.

        Args:
            img_path: Path to the image file.

        Returns:
            embeddings:  (K, D) float tensor of normalised pixel embeddings
                         where K = orig_width × orig_height, in column-major
                         (x, y) order identical to PixelEmbeddingGenerator.
            xy_to_index: Dict mapping every pixel (x, y) in the original image
                         to its flat index in the embeddings tensor.
        """
        start = time.perf_counter()

        # ------------------------------------------------------------------
        # Step 1: Load and pad
        # ------------------------------------------------------------------
        pil_image = load_image(img_path)
        orig_w, orig_h = pil_image.size

        padded_image = pad_to_multiple(pil_image, self.patch_size)
        padded_w, padded_h = padded_image.size

        grid_cols = padded_w // self.patch_size
        grid_rows = padded_h // self.patch_size
        K_patch = grid_rows * grid_cols

        logger.debug(
            f"Image '{img_path}': original={orig_w}x{orig_h}, "
            f"padded={padded_w}x{padded_h}, grid={grid_rows}x{grid_cols}"
        )

        # ------------------------------------------------------------------
        # Step 2: Single forward pass — hidden states are (num_tokens, D) on CPU
        # ------------------------------------------------------------------
        t_model = time.perf_counter()
        hidden = self.model.generate_embedding(padded_image)   # (num_tokens, D)
        patch_tokens = hidden[-K_patch:, :]                    # (K_patch, D)
        D = patch_tokens.shape[1]
        logger.debug(f"Forward pass: {time.perf_counter() - t_model:.3f}s")

        # ------------------------------------------------------------------
        # Step 3: Reshape for F.interpolate — expects (N, C, H, W)
        #
        # ViT patch tokens are in row-major order, so reshaping to
        # (grid_rows, grid_cols, D) is correct before permuting to (D, H, W).
        # ------------------------------------------------------------------
        grid = (
            patch_tokens
            .reshape(grid_rows, grid_cols, D)   # (grid_rows, grid_cols, D)
            .permute(2, 0, 1)                   # (D, grid_rows, grid_cols)
            .unsqueeze(0)                        # (1, D, grid_rows, grid_cols)
        )

        # ------------------------------------------------------------------
        # Step 4: Bilinear upsample to the padded image size.
        #
        # Upsampling to (padded_h, padded_w) rather than (orig_h, orig_w)
        # preserves correct alignment: with align_corners=False, pixel p maps
        # to input coordinate (p + 0.5) * (grid_size / padded_size) - 0.5
        #                   = (p + 0.5) / patch_size - 0.5.
        # At p = k * patch_size + patch_size/2 - 0.5 (patch k's centre) this
        # evaluates to k — the exact integer grid index. ✓
        # ------------------------------------------------------------------
        t_interp = time.perf_counter()
        upsampled = F.interpolate(
            grid,
            size=(padded_h, padded_w),
            mode="bilinear",
            align_corners=False,
        )   # (1, D, padded_h, padded_w)
        logger.debug(f"Interpolation: {time.perf_counter() - t_interp:.3f}s")

        # ------------------------------------------------------------------
        # Step 5: Crop to original image dimensions (discard padding region)
        # ------------------------------------------------------------------
        cropped = upsampled[:, :, :orig_h, :orig_w]   # (1, D, orig_h, orig_w)

        # ------------------------------------------------------------------
        # Step 6: Reshape to (K, D) with column-major flat_idx = x * orig_h + y
        #
        # cropped.squeeze(0) → (D, orig_h, orig_w)  dims: (D, y, x)
        # .permute(2, 1, 0)  → (orig_w, orig_h, D)  dims: (x, y, D)
        # .reshape(K, D)     → flat_idx = x * orig_h + y  ✓  (matches pixel.py)
        # ------------------------------------------------------------------
        embeddings = (
            cropped.squeeze(0)              # (D, orig_h, orig_w)
            .permute(2, 1, 0)              # (orig_w, orig_h, D)
            .reshape(orig_w * orig_h, D)   # (K, D)
        )

        # ------------------------------------------------------------------
        # Step 7: Normalise
        # ------------------------------------------------------------------
        if self.normalize:
            embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)
        embeddings = F.normalize(embeddings, p=2, dim=1, eps=1e-8)

        # ------------------------------------------------------------------
        # Step 8: xy_to_index — column-major, matching pixel.py exactly
        # ------------------------------------------------------------------
        xy_to_index = LazyXYMapping("colmajor", orig_w, orig_h)

        elapsed = time.perf_counter() - start
        logger.debug(
            f"Bilinear embedding for '{img_path}': K={embeddings.shape[0]}, "
            f"D={D}, time={elapsed:.3f}s"
        )

        return embeddings, xy_to_index
