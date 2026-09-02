from __future__ import annotations
"""
Utility functions for nonconformity score generation.

This module provides helpers used by the score generator classes for:
- Extracting landmark embeddings and indices from xy_to_index maps
- Converting ragged score arrays into padded matrices for cp4icl
- Removing self-scores for leave-one-out calibration methods
- Memory-efficient GPU softmax for large embedding pairs
- Memory estimation for large tensor allocations
"""

import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


def get_landmark_indices(landmarks: list[list[float]], xy_maps: list[dict]) -> list[int]:
    """
    Look up the flat embedding index for each landmark coordinate.

    Args:
        landmarks: List of [x, y] coordinates, one per image.
        xy_maps: List of dicts mapping (x, y) -> flat index, one per image.

    Returns:
        List of integer indices into each image's (K, D) embedding tensor.
    """
    indices = []
    for lm, xy_map in zip(landmarks, xy_maps):
        x, y = int(lm[0]), int(lm[1])
        indices.append(xy_map[(x, y)])
    return indices


def extract_landmark_embeddings(landmarks: list[list[float]], embeddings: list[torch.Tensor], xy_maps: list[dict]) -> list[torch.Tensor]:
    """
    Extract the landmark embedding vector from each image's full embedding tensor.

    Each returned tensor is a clone to ensure it remains valid even if the
    full embedding tensors are later freed.

    Args:
        landmarks: List of [x, y] coordinates, one per image.
        embeddings: List of (K_i, D) tensors, one per image.
        xy_maps: List of dicts mapping (x, y) -> flat index, one per image.

    Returns:
        List of (D,) tensors, one landmark embedding per image.
    """
    lm_embeddings = []
    for lm, emb, xy_map in zip(landmarks, embeddings, xy_maps):
        x, y = int(lm[0]), int(lm[1])
        idx = xy_map[(x, y)]
        lm_embeddings.append(emb[idx].clone())
    return lm_embeddings


def scores_to_matrix(score_list: list[np.ndarray], pad_value: float = 1.1) -> np.ndarray:
    """
    Convert a list of ragged score arrays into a padded (N, M, K_max) matrix.

    Each element in score_list is a (K_m, N) array of nonconformity scores
    for test image m across N calibration sources. Since K_m varies across
    test images, shorter arrays are padded with `pad_value`.

    Args:
        score_list: List of M arrays, each of shape (K_m, N).
        pad_value: Value to fill for padded positions (should be > 1.0
            so padded entries are never included in prediction sets).

    Returns:
        Padded matrix of shape (N, M, K_max).
    """
    M = len(score_list)
    N = score_list[0].shape[1]
    K_max = max(s.shape[0] for s in score_list)

    matrix = np.full((N, M, K_max), pad_value)
    for m in range(M):
        K_m = score_list[m].shape[0]
        matrix[:, m, :K_m] = score_list[m].T  # (K_m, N).T → (N, K_m)

    logger.debug(f"Score matrix: (N={N}, M={M}, K_max={K_max})")
    return matrix

def remove_self_2d(mat: np.ndarray) -> np.ndarray:
    """
    Remove self-scores from the diagonal of an (N, N) calibration matrix.

    Used by SCOS and model-selection methods which require leave-one-out
    calibration scores.

    Args:
        mat: (N, N) array where mat[i, j] is the score of source i on
            calibration target j.

    Returns:
        (N, N-1) array with the diagonal entries removed from each row.
    """
    N = mat.shape[0]
    mask = ~np.eye(N, dtype=bool)
    return mat[mask].reshape(N, N - 1)


def remove_self_3d(tensor: np.ndarray) -> np.ndarray:
    """
    Remove self-scores from the diagonal of an (N, N, K) calibration tensor.

    Extends remove_self_2d to the all-label case where each (i, j) entry
    contains K candidate scores rather than a single true-label score.

    Args:
        tensor: (N, N, K) array.

    Returns:
        (N, N-1, K) array with the diagonal slices removed.
    """
    N = tensor.shape[0]
    mask = ~np.eye(N, dtype=bool)
    # mask is (N, N), broadcast over K dimension
    return tensor[mask].reshape(N, N - 1, tensor.shape[2])


# For now not using it
@torch.inference_mode()
def softmax_prob_true_column(
    test_gpu: torch.Tensor,
    calib_gpu: torch.Tensor,
    true_idx: int,
    temperature: float,
    chunk_k: int = 8192,
    chunk_q: int = 4096,
) -> torch.Tensor:
    """
    Compute softmax probability of the true calibration pixel for each test pixel,
    without materializing the full [K_test, K_calib] similarity matrix.

    This is used for generating reverse scores at pixel level, where K_test and
    K_calib can each be ~300K+, making the full matrix (~360 GB at float32)
    impossible to allocate.

    The computation is equivalent to:
        logits = (test_gpu @ calib_gpu.T) / temperature    # [K_t, K_i]
        probs = softmax(logits, dim=1)                     # [K_t, K_i]
        return probs[:, true_idx]                          # [K_t]

    but done in chunks using the log-sum-exp identity to keep peak memory bounded
    by chunk_q * chunk_k floats.

    Args:
        test_gpu: (K_test, D) test embeddings on GPU.
        calib_gpu: (K_calib, D) calibration embeddings on GPU.
        true_idx: Column index of the true calibration label.
        temperature: Softmax temperature.
        chunk_k: Chunk size along the calibration (key) dimension.
        chunk_q: Chunk size along the test (query) dimension.

    Returns:
        (K_test,) float32 tensor of p_true values on GPU.
    """
    T = temperature
    K_t = test_gpu.shape[0]
    device = test_gpu.device

    # Score of the true label for all test pixels
    k_true = calib_gpu[true_idx : true_idx + 1, :]  # (1, D)
    s_true = (test_gpu @ k_true.T).squeeze(1) / T   # (K_t,)

    # Compute log-sum-exp over all calib pixels in chunks
    lse_all = torch.empty(K_t, device=device, dtype=torch.float32)

    for qs in range(0, K_t, chunk_q):
        qe = min(qs + chunk_q, K_t)
        q = test_gpu[qs:qe]  # (Bq, D)

        # Running logsumexp initialized to -inf
        lse = torch.full(
            (q.shape[0],), float("-inf"), device=device, dtype=torch.float32
        )

        for ks in range(0, calib_gpu.shape[0], chunk_k):
            ke = min(ks + chunk_k, calib_gpu.shape[0])
            k = calib_gpu[ks:ke]  # (Bk, D)

            logits = (q @ k.T) / T  # (Bq, Bk)
            lse = torch.logaddexp(lse, torch.logsumexp(logits.float(), dim=1))
            del logits

        lse_all[qs:qe] = lse
        del q, lse

    p_true = torch.exp(s_true.float() - lse_all)  # (K_t,)
    return p_true

def estimate_memory_gb(shape: tuple, dtype=np.float64) -> float:
    """
    Estimate memory in GB for a numpy array of the given shape and dtype.

    Useful for logging warnings before allocating large matrices
    (e.g., the (N, N, K_max) calibration all-scores tensor).

    Args:
        shape: Tuple of dimension sizes.
        dtype: Numpy dtype (default float64).

    Returns:
        Estimated memory in GB.
    """
    bytes_per = np.dtype(dtype).itemsize
    total_elements = 1
    for s in shape:
        total_elements *= s
    return total_elements * bytes_per / 1e9