"""
Fused distance scoring ("Scheme B").

What a "score generator" is:
    One of the pipeline's interchangeable scoring strategies (alongside
    per_source_distance.py and cosine_softmax.py). It turns the embeddings into a
    number for every candidate location — roughly "how unlikely is this spot to be
    the landmark" — and the conformal-prediction (CP) stage later thresholds those
    numbers into a prediction set.

How this strategy scores:
    per_source_distance.py lets every source vote with its own peak and combines
    the votes afterward. This file does the opposite order: it COMBINES the sources
    first into one agreed-upon map, picks the single best spot on it (the "fused
    peak"), and then scores every candidate by its distance to that one spot.
    Because there's exactly one centre, the kept region comes out as one clean
    shape (a single Manhattan diamond) instead of a fuzzy blob of overlapping votes.

The one gotcha — the CP stage must use k=1:
    Combining the sources is normally the CP stage's job. We've already done it in
    here, so the CP stage must NOT combine again — if it does, you aggregate twice
    and the scores are wrong. So this scoring mode only works with CAOS run at k=1.
    We don't set that here; cp_k_for_score() is the single place that forces it.
    Watch out for the two different k's: the "fusion k" we pass to this class (how
    many sources to average when building the fused map) is unrelated to the
    CP-stage k (which is pinned to 1).
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


class FusedDistanceScoreGenerator:
    """
    Scores each candidate by its distance to ONE fused peak ("Scheme B").

    Where it fits:
        Same plug-in contract as the other score generators — the pipeline calls
        generate_calib_scores(...) and generate_eval_scores(...), and this returns
        arrays caos_ragged can consume. The catch is the k=1 requirement below.

    What it does, step by step:
        1. Every source compares its landmark embedding against the candidates to
           get a similarity map.
        2. We merge all those maps into ONE map (keep each candidate's k
           best-agreeing sources and average them — see _fused_peak).
        3. The fused peak is the single best spot on that merged map.
        4. A candidate's score is its distance to that one fused peak.

    Contrast with per_source_distance.py ("Scheme A"), which keeps one peak per
    source and lets the CP stage combine the distances. Here we combine in step 2,
    before any distance is measured.

    The k=1 requirement (important):
        Because we already merged the sources, there's nothing left for the CP
        stage to merge, so it must run with k=1 (a no-op pass-through). Don't set
        that here — cp_k_for_score() enforces it. The `k` this class takes is the
        FUSION k (how many sources to average in step 2), which is separate from
        the CP-stage k.

    Why the shapes look "degenerate":
        Since all sources collapse into one before scoring, only a single "source"
        is left. So calibration returns (1, N) and evaluation returns (K_m, 1) —
        that size-1 axis is the leftover single fused source, which caos_ragged
        (k=1) passes straight through.

    Why conformal prediction stays valid:
        Calibration fuses the OTHER sources (leave-one-out) and measures
        distance(fused peak, true landmark); evaluation fuses all sources and
        measures distance(fused peak, candidate). Same function, true point swapped
        for a candidate — the symmetry CP needs.

    Args:
        metric (str): "manhattan", "euclidean", or "chebyshev".
        k (int): fusion k — how many best-agreeing sources to average when building
            the fused map.
        device (str): torch device.
        verbose (bool): show progress bars and timing logs.
    """

    def __init__(self, metric: str = "manhattan", k: int = 3,
                 device: str = "cuda", verbose: bool = True):
        if metric not in _METRICS:
            raise ValueError(f"metric must be one of {_METRICS}, got {metric!r}")
        self.metric = metric
        self.k = k
        self.device = device
        self.verbose = verbose

    # ── distance primitives ──────────────────────────────────────────────────
    def _distance(self, dx: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
        """Turn per-axis gaps |dx|, |dy| into a single distance, per the chosen metric."""
        if self.metric == "manhattan":
            return dx + dy
        if self.metric == "euclidean":
            return torch.sqrt(dx * dx + dy * dy)
        return torch.maximum(dx, dy)  # chebyshev

    def _dist_point_to_coords(self, point: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Distance from a single (2,) point to every one of (K, 2) coords -> (K,)."""
        dx = (coords[:, 0] - point[0]).abs()
        dy = (coords[:, 1] - point[1]).abs()
        return self._distance(dx, dy)

    def _fused_peak(self, cos_sims: torch.Tensor, coords: torch.Tensor, k: int) -> torch.Tensor:
        """Merge the sources' (S, K) similarity maps into one and return its best
        spot's (x, y).

        First we flip cosine similarity into "badness" (1 - cos, so lower is
        better, matching the rest of the pipeline). Then at each candidate we keep
        its k best-agreeing sources and average them. The fused peak is the
        candidate with the lowest averaged badness — the spot the top-k sources
        most agree on.
        """
        nonconf = 1.0 - cos_sims                                    # (S, K)
        kk = min(k, nonconf.shape[0])
        topk_vals, _ = torch.topk(nonconf, k=kk, dim=0, largest=False)  # (kk, K)
        fused = topk_vals.mean(dim=0)                               # (K,)
        return coords[int(fused.argmin())]                         # (2,)

    # ── calibration ──────────────────────────────────────────────────────────
    def generate_calib_scores(self, embeddings: list[torch.Tensor], landmarks: list[list[float]],
                              xy_maps: list, return_all_scores: bool = False,
                              return_ragged: bool = False) -> np.ndarray:
        """
        Score the calibration set (leave-one-out), one fused peak per image.

        For each calibration image j, fuse every OTHER source on that image into a
        single peak, then record how far that peak is from image j's true landmark.
        (We leave source j out so an image never helps score itself.)

        Returns:
            (1, N) array — one fused distance-to-truth per calibration image. The
            leading size-1 axis is the single fused source (see the class docstring).
        """
        if return_all_scores or return_ragged:
            raise NotImplementedError(
                "return_all_scores/return_ragged are not supported for "
                "FusedDistanceScoreGenerator (only CAOS with k=1 is targeted)."
            )

        start = time.perf_counter()
        N = len(embeddings)
        lm_embeddings = extract_landmark_embeddings(landmarks, embeddings, xy_maps)
        calib_scores = np.empty((1, N), dtype=np.float64)

        with torch.no_grad():
            Q = torch.stack(lm_embeddings, dim=0).to(self.device)   # (N, D)
            gt_coords = torch.tensor(
                [[int(lm[0]), int(lm[1])] for lm in landmarks],
                dtype=torch.float32, device=self.device,
            )                                                        # (N, 2)

            iterator = (
                tqdm(range(N), desc="Calibration fused distances", leave=False)
                if self.verbose else range(N)
            )
            for j in iterator:
                E_j = embeddings[j].to(self.device)                 # (K_j, D)
                coords_j = torch.as_tensor(
                    xy_maps[j].coords(), dtype=torch.float32, device=self.device
                )                                                    # (K_j, 2)

                cos_sims = Q @ E_j.T                                # (N, K_j)
                # leave-one-out: drop source j before fusing on image j.
                keep = torch.arange(N, device=self.device) != j
                peak = self._fused_peak(cos_sims[keep], coords_j, self.k)   # (2,)
                calib_scores[0, j] = self._dist_point_to_coords(
                    peak, gt_coords[j:j + 1]
                ).item()

                del E_j, coords_j, cos_sims, peak

        elapsed = time.perf_counter() - start
        logger.info(f"Calibration fused distances (1x{N}, metric={self.metric}, "
                    f"fuse_k={self.k}): {elapsed:.2f}s")
        return calib_scores

    # ── evaluation ───────────────────────────────────────────────────────────
    def generate_eval_scores(self, test_embeddings: list[torch.Tensor],
                             calib_lm_embeddings: list[torch.Tensor],
                             xy_maps: list) -> list[np.ndarray]:
        """
        Score every candidate on every test image against the fused peak.

        For each test image, fuse all the sources into one peak, then measure every
        candidate's distance to that single peak.

        Returns:
            One (K_m, 1) array per test image: row = candidate, the single column is
            its distance to the fused peak. caos_ragged (k=1) consumes it directly.
        """
        start = time.perf_counter()
        M = len(test_embeddings)
        all_scores = []

        with torch.no_grad():
            Q = torch.stack(calib_lm_embeddings, dim=0).to(self.device)  # (N, D)

            iterator = (
                tqdm(range(M), desc="Evaluation fused distances", leave=False)
                if self.verbose else range(M)
            )
            for m in iterator:
                T_m = test_embeddings[m].to(self.device)            # (K_m, D)
                coords_m = torch.as_tensor(
                    xy_maps[m].coords(), dtype=torch.float32, device=self.device
                )                                                    # (K_m, 2)

                cos_sims = Q @ T_m.T                                # (N, K_m)
                peak = self._fused_peak(cos_sims, coords_m, self.k)  # (2,)
                dist = self._dist_point_to_coords(peak, coords_m)   # (K_m,)

                all_scores.append(dist.cpu().numpy().reshape(-1, 1))  # (K_m, 1)
                del T_m, coords_m, cos_sims, peak, dist

        elapsed = time.perf_counter() - start
        logger.info(f"Evaluation fused distances ({M} test images, "
                    f"metric={self.metric}, fuse_k={self.k}): {elapsed:.2f}s")
        return all_scores
