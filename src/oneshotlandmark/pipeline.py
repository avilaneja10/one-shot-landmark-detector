"""
Pipeline for one-shot landmark conformal prediction.
 
Orchestrates embedding generation, cosine similarity computation, and
nonconformity scoring with optional caching at each stage. Scripts
use the pipeline instead of calling generators directly.
 
Caching strategy:
  - Embeddings: cached as raw patch tokens (post-normalization). This
    has been kind of mandate but TODO : Make it also variable in terms of caching.
  - Cosine similarities: cached as raw Q @ candidates.T (pre-temperature,
    pre-softmax) so that temperature sweeps are instant replays.
    For every landmark this will be stored.
  - Reverse scores: NOT cached at cosine level (too large at pixel level).
    Computed directly from embeddings each time.
"""

import time
import logging
 
import torch
import numpy as np
 
from oneshotlandmark.model import ViTModel
from oneshotlandmark.embeddings.patch import PatchEmbeddingGenerator
from oneshotlandmark.embeddings.pixel import PixelEmbeddingGenerator
from oneshotlandmark.scores.generator import ScoreGenerator
from oneshotlandmark.cache.base import BaseCache

logger = logging.getLogger(__name__)

class Pipeline:
    def __init__(self, level: str = "patch", landmark_idx: int = 0, model: ViTModel = None, patch_size: int = 16,
                device: str = "cuda", cache: BaseCache = None, dataset_id: str = "default", verbose: bool = True):
        self.level = level
        self.landmark_idx = landmark_idx
        self.patch_size = patch_size
        self.device = device
        self.cache = cache
        self.dataset_id = dataset_id
        self.verbose = verbose
 
        if model is None:
            model = ViTModel(device_str=device)
        self.model = model
 
        if level == "patch":
            self.emb_gen = PatchEmbeddingGenerator(
                model=model, patch_size=patch_size, verbose=verbose,
            )
        elif level == "pixel":
            self.emb_gen = PixelEmbeddingGenerator(
                model=model, patch_size=patch_size, verbose=verbose,
            )
        else:
            raise ValueError(f"Unknown level: {level}. Must be 'patch' or 'pixel'.")
 
        self.score_gen = ScoreGenerator(device=device, verbose=verbose, level=level, patch_size=patch_size)

    # ========================
    # CACHING KEY HELPERS
    # ========================
    # TODO : Since for each dataset we can have various images
    # Currently they will be mapped to same key but we need to design
    # a strategy to cache based on images rather than dataset id
    def _embedding_key(self, split: str) -> str:
        return f"{self.dataset_id}_{split}_embeddings_{self.level}_ps{self.patch_size}"
 
    def _cosine_key(self, kind: str) -> str:
        return (
            f"{self.dataset_id}_{kind}_cosines_{self.level}"
            f"_ps{self.patch_size}_lm{self.landmark_idx}"
        )

    # ========================
    # CACHING EXISTS HELPERS
    # ========================
    def cosines_cached(self) -> bool:
        """Check if both calib and eval cosines are already cached."""
        if not self.cache:
            return False
        return (
            self.cache.exists(self._cosine_key("calib"))
            and self.cache.exists(self._cosine_key("eval"))
        )

    def embeddings_cached(self, split: str) -> bool:
        """Check if embeddings for the given split are already cached."""
        if not self.cache:
            return False
        return self.cache.exists(self._embedding_key(split))

    
    def get_embeddings(self, img_paths, split):
        """
        Get embeddings for a list of images, using cache if available.
 
        Args:
            img_paths: List of image file paths.
            split: Name for this data split (e.g., "calib", "test").
 
        Returns:
            embeddings: List of (K_i, D) tensors.
        """
        emb_key = self._embedding_key(split)

        # Try to load embeddings from the cache if exists
        if self.cache and self.cache.exists(emb_key):
            embeddings = self.cache.load(emb_key)
            return embeddings
 
        logger.info(f"Computing embeddings for {split} ({len(img_paths)} images)")
        embeddings = self.emb_gen.generate_embedding_all(img_paths)
 
        if self.cache:
            self.cache.save(emb_key, embeddings)        # 25 GB — loaded only when needed
 
        return embeddings
    
    def get_calib_cosines(self, calib_embeddings=None, calib_landmarks=None, calib_img_dims=None):
        """
        Get raw calibration cosine similarities, using cache if available.
 
        Args:
            calib_embeddings: List of N tensors, each (K_j, D).
            calib_landmarks: List of N [x, y] coordinates.
 
        Returns:
            Dict with "cosines" (list of N arrays) and "gt_indices" (list of N ints).
        """
        key = self._cosine_key("calib")
 
        if self.cache and self.cache.exists(key):
            logger.info(f"Loading cached calib cosines: {key}")
            return self.cache.load(key)
 
        logger.info("Computing calibration cosines")
        cosines = self.score_gen.compute_calib_cosines(
            calib_embeddings, calib_landmarks, calib_img_dims
        )
 
        if self.cache:
            self.cache.save(key, cosines)
 
        return cosines

    def get_eval_cosines(self, test_embeddings=None, calib_lm_embeddings=None):
        """
        Get raw evaluation cosine similarities, using cache if available.
 
        Args:
            test_embeddings: List of M tensors, each (K_m, D).
            calib_lm_embeddings: List of N tensors, each (D,) — from
                get_calib_cosines result["lm_embeddings"].
 
        Returns:
            Dict with "cosines" (list of M arrays, each (N, K_m)).
        """
        key = self._cosine_key("eval")
 
        if self.cache and self.cache.exists(key):
            logger.info(f"Loading cached eval cosines: {key}")
            return self.cache.load(key)
 
        logger.info("Computing evaluation cosines")
        cosines = self.score_gen.compute_eval_cosines(
            test_embeddings, calib_lm_embeddings
        )
 
        if self.cache:
            self.cache.save(key, cosines)
 
        return cosines

    def get_calib_scores(self, calib_cosines, temperature=1.0, apply_softmax=True,return_all_scores=False):
        """Apply temperature and softmax to cached calibration cosines."""
        logger.info(f"Applying calib scoring: temp={temperature}, softmax={apply_softmax}")
        return ScoreGenerator.apply_calib_scoring(
            calib_cosines,
            temperature=temperature,
            apply_softmax=apply_softmax,
            return_all_scores=return_all_scores,
        )
 
    def get_eval_scores(self, eval_cosines, temperature=1.0, apply_softmax=True):
        """Apply temperature and softmax to cached evaluation cosines."""
        logger.info(f"Applying eval scoring: temp={temperature}, softmax={apply_softmax}")
        return ScoreGenerator.apply_eval_scoring(
            eval_cosines,
            temperature=temperature,
            apply_softmax=apply_softmax,
        )

    # TODO : Add support for reverse scores too for fullcaos