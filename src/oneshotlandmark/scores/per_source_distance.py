"""
Per-source distance scoring ("Scheme A").

What a "score generator" is:
    The pipeline has a few interchangeable scoring strategies (this one,
    fused_distance.py, and cosine_softmax.py). Each one takes the image
    embeddings and produces a number for every candidate location — think of it
    as "how unlikely is this spot to be the landmark". The conformal-prediction
    (CP) stage later picks a threshold on those numbers and keeps everything
    below it as the prediction set.

How this strategy scores:
    A "source" is one labeled example. On a given image, a source produces a
    similarity map, and the single hottest spot on that map is its "peak" — the
    source's guess for where the landmark is. We score a candidate simply by how
    far it sits from that peak. Close to the peak -> small score -> more likely
    to be kept.

    Every source keeps its OWN peak, so each candidate ends up with one distance
    per source. We don't combine them here — we hand all the per-source distances
    to the CP stage and let it do the combining.

The sibling file:
    fused_distance.py ("Scheme B") merges all the sources into one peak FIRST and
    measures distance to that single spot. Same idea, opposite order. That order
    is the only real difference between the two.
"""
from __future__ import annotations
import torch
import numpy as np
import time
import logging
from tqdm import tqdm
from oneshotlandmark.scores.utils import extract_landmark_embeddings

logger = logging.getLogger(__name__)

_METRICS = {"manhattan", "euclidean", "chebyshev"}


class PerSourceDistanceScoreGenerator:
    """
    Scores each candidate by its distance to every source's peak ("Scheme A").

    Where it fits:
        Drop-in alternative to CosineSoftmaxScoreGenerator. The pipeline calls the
        same two methods on any score generator — generate_calib_scores(...) to
        score the calibration set and generate_eval_scores(...) to score the test
        candidates — and this returns arrays in the exact same shapes, so nothing
        downstream (the CP methods, the coverage / set-size code) has to change.

    What it does, step by step:
        1. Each source has a landmark embedding. Compare it against every candidate
           on an image to get a similarity map.
        2. The peak is the argmax of that map — the source's single best guess.
        3. A candidate's score is the distance from that peak to the candidate
           (Manhattan / Euclidean / Chebyshev — your choice via `metric`).
        4. Do this for all sources, so each candidate collects one distance per
           source. The CP stage combines across sources later; we don't here.

    Where the coordinates come from:
        We need real (x, y) positions to measure distance. LazyXYMapping.coords()
        gives the (x, y) of every embedding row in the original image. For patch
        embeddings it returns the patch centre, so patch and pixel distances are on
        the same scale.

    Why conformal prediction stays valid:
        Calibration measures distance(peak, true landmark); evaluation measures
        distance(peak, candidate). It's the same function with the true point
        swapped for a candidate, which is the symmetry CP relies on.

    Args:
        metric (str): "manhattan", "euclidean", or "chebyshev".
        device (str): torch device, e.g. "cuda" or "cpu".
        verbose (bool): show progress bars and timing logs.
    """

    def __init__(self, metric: str = "manhattan", device: str = "cuda", verbose: bool = True):
        if metric not in _METRICS:
            raise ValueError(f"metric must be one of {_METRICS}, got {metric!r}")
        self.metric = metric
        self.device = device
        self.verbose = verbose

    # ── distance primitives ──────────────────────────────────────────────────
    def _distance(self, dx: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
        """Turn per-axis gaps |dx|, |dy| into a single distance, per the chosen
        metric. Works on any broadcastable shapes."""
        if self.metric == "manhattan":
            return dx + dy
        if self.metric == "euclidean":
            return torch.sqrt(dx * dx + dy * dy)
        # chebyshev
        return torch.maximum(dx, dy)

    def _point_distance(self, peaks: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
        """Distance from each of (N, 2) peaks to a single (2,) point -> (N,)."""
        dx = (peaks[:, 0] - point[0]).abs()
        dy = (peaks[:, 1] - point[1]).abs()
        return self._distance(dx, dy)

    def _pairwise_distance(self, peaks: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Distance from each of (N, 2) peaks to every one of (K, 2) coords -> (N, K).

        We broadcast one axis at a time so we never build the full (N, K, 2)
        tensor: dx is (N, 1) - (1, K) -> (N, K), and likewise dy.
        """
        dx = (peaks[:, 0:1] - coords[:, 0].unsqueeze(0)).abs()  # (N, K)
        dy = (peaks[:, 1:2] - coords[:, 1].unsqueeze(0)).abs()  # (N, K)
        return self._distance(dx, dy)

    def _peak_coords(self, cos_sims: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Find each source's peak: take the argmax of its (N, K) similarity map
        and look up that cell's (x, y) -> (N, 2)."""
        peak_idx = cos_sims.argmax(dim=1)  # (N,)
        return coords[peak_idx]            # (N, 2)

    # ── calibration ──────────────────────────────────────────────────────────
    def generate_calib_scores(self, embeddings: list[torch.Tensor], landmarks: list[list[float]],
                              xy_maps: list, return_all_scores: bool = False,
                              return_ragged: bool = False) -> np.ndarray:
        """
        Score the calibration set (leave-one-out).

        For each calibration image j, every source finds its peak on image j, and
        we record how far that peak is from image j's true landmark. A source is
        never scored against its own image — those self-entries on the diagonal are
        set to infinity so CP ignores them.

        Args:
            embeddings: N tensors, each (K_j, D) — one per calibration image.
            landmarks: N [x, y] ground-truth coordinates.
            xy_maps: N LazyXYMapping objects (we call .coords() on each).
            return_all_scores / return_ragged: only the model-selection (yk_*)
                methods need these, and distance scoring doesn't support them, so
                asking for either raises.

        Returns:
            (N, N) array. Entry [i, j] = distance from source i's peak on image j
            to image j's true landmark. Diagonal = inf. These are the calibration
            scores CP calibrates its threshold from.
        """
        if return_all_scores or return_ragged:
            raise NotImplementedError(
                "return_all_scores/return_ragged are not supported for "
                "PerSourceDistanceScoreGenerator (only CAOS/SCOS are targeted)."
            )

        start = time.perf_counter()
        N = len(embeddings)

        lm_embeddings = extract_landmark_embeddings(landmarks, embeddings, xy_maps)

        calib_scores = np.empty((N, N), dtype=np.float64)

        with torch.no_grad():
            Q = torch.stack(lm_embeddings, dim=0).to(self.device)  # (N, D)
            gt_coords = torch.tensor(
                [[int(lm[0]), int(lm[1])] for lm in landmarks],
                dtype=torch.float32, device=self.device,
            )  # (N, 2)

            iterator = (
                tqdm(range(N), desc="Calibration distances", leave=False)
                if self.verbose else range(N)
            )

            for j in iterator:
                E_j = embeddings[j].to(self.device)                         # (K_j, D)
                coords_j = torch.as_tensor(
                    xy_maps[j].coords(), dtype=torch.float32, device=self.device
                )                                                            # (K_j, 2)

                cos_sims = Q @ E_j.T                                        # (N, K_j)
                peaks = self._peak_coords(cos_sims, coords_j)              # (N, 2)
                calib_scores[:, j] = self._point_distance(
                    peaks, gt_coords[j]
                ).cpu().numpy()                                            # (N,)

                del E_j, coords_j, cos_sims, peaks

        np.fill_diagonal(calib_scores, np.inf)

        elapsed = time.perf_counter() - start
        logger.info(f"Calibration distances ({N}x{N}, metric={self.metric}): {elapsed:.2f}s")
        return calib_scores

    # ── evaluation ───────────────────────────────────────────────────────────
    def generate_eval_scores(self, test_embeddings: list[torch.Tensor],
                             calib_lm_embeddings: list[torch.Tensor],
                             xy_maps: list) -> list[np.ndarray]:
        """
        Score every candidate on every test image.

        For each test image, every source finds its peak, and we then measure the
        distance from that peak to every candidate location on the image.

        Args:
            test_embeddings: M tensors, each (K_m, D) — one per test image.
            calib_lm_embeddings: N calibration landmark embeddings, each (D,).
            xy_maps: M LazyXYMapping objects for the test images (we call .coords()
                for candidate coordinates). This is the one extra argument compared
                to CosineSoftmaxScoreGenerator.

        Returns:
            One (K_m, N) array per test image m: row = candidate, column = source,
            value = distance from that source's peak to that candidate. caos_ragged
            consumes this directly.
        """
        start = time.perf_counter()
        M = len(test_embeddings)
        all_scores = []

        with torch.no_grad():
            Q = torch.stack(calib_lm_embeddings, dim=0).to(self.device)  # (N, D)

            iterator = (
                tqdm(range(M), desc="Evaluation distances", leave=False)
                if self.verbose else range(M)
            )

            for m in iterator:
                T_m = test_embeddings[m].to(self.device)                   # (K_m, D)
                coords_m = torch.as_tensor(
                    xy_maps[m].coords(), dtype=torch.float32, device=self.device
                )                                                          # (K_m, 2)

                cos_sims = Q @ T_m.T                                       # (N, K_m)
                peaks = self._peak_coords(cos_sims, coords_m)             # (N, 2)
                dist = self._pairwise_distance(peaks, coords_m)           # (N, K_m)

                all_scores.append(dist.T.cpu().numpy())                   # (K_m, N)

                del T_m, coords_m, cos_sims, peaks, dist

        elapsed = time.perf_counter() - start
        logger.info(
            f"Evaluation distances ({M} test images, metric={self.metric}): {elapsed:.2f}s"
        )
        return all_scores
