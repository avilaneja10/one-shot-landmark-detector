from __future__ import annotations
from collections.abc import Mapping
import numpy as np

class LazyXYMapping(Mapping):
    """
    Lazy, formula-based replacement for the (x, y) -> flat-index dict that the
    embedding generators used to return.

    WHY THIS EXISTS
    ---------------
    Each embedding generator (`patch.py`, `bilinear_interpolation.py`,
    `pixel.py`) returns a `(K, D)` embedding tensor together with a map from
    every original-image pixel `(x, y)` to the flat row in that tensor. At
    pixel/bilinear level the key space is the whole image (`orig_w * orig_h`,
    ~200K+ entries), so materializing this as a real dict — for all N calib +
    M test images at once — costs a lot of time to build and a lot of RAM to
    hold, while every entry is just a closed-form function of `(x, y)`.

    This class stores only the handful of parameters (image size, patch size,
    grid shape) and computes the index on demand, so the map is effectively
    free to construct and hold. It subclasses `collections.abc.Mapping` and
    reproduces the old dict's read semantics exactly (same index formulas,
    same KeyError only for out-of-image coordinates), so it is a drop-in with
    no contract change.

    WHERE IT IS PRODUCED
    --------------------
    Returned as the second element of `generate_embedding(...)` /
    `generate_embedding_all(...)` by all three generators, replacing the dict:
      - patch.py                -> layout="patch"
      - bilinear_interpolation.py, pixel.py -> layout="colmajor"

    HOW IT IS CONSUMED
    ------------------
    - `scores/utils.py::get_landmark_indices` and `extract_landmark_embeddings`
      do `xy_map[(x, y)]` to turn a ground-truth landmark pixel into the flat
      embedding row. This is the *only* operation the existing pipeline uses.
    - `coords()` is the inverse (embedding row -> pixel `(x, y)`), added for the
      distance/regression scorer (`DistanceScoreGenerator`), which needs the
      pixel coordinate of every candidate to compute distance-to-peak scores.
      For patch layout it returns patch-CENTER pixel coordinates, which keeps
      patch and pixel prediction-set radii in the same original-pixel frame.

    IMPORTANT DISTINCTION (key space vs. index space)
    -------------------------------------------------
    `__len__` / `__iter__` describe the KEY space — every pixel, `orig_w*orig_h`
    for ALL layouts (patch keyed every pixel too, just mapping many pixels to
    the same patch index). The number of embedding rows `K` is a DIFFERENT
    quantity: `grid_rows*grid_cols` at patch level, `orig_w*orig_h` otherwise.
    So never use `len(xy_map)` as `K` — read `embeddings.shape[0]` (or
    `coords().shape[0]`) instead; they diverge at patch level.

    layout="colmajor": idx = x*orig_h + y                  (pixel, bilinear)
    layout="patch":    idx = (y//ps)*grid_cols + (x//ps)   (patch)
    """
    def __init__(self, layout, orig_w, orig_h,
                 patch_size=None, grid_cols=None, grid_rows=None):
        self.layout, self.orig_w, self.orig_h = layout, orig_w, orig_h
        self.ps, self.grid_cols, self.grid_rows = patch_size, grid_cols, grid_rows

    def __getitem__(self, key):
        x, y = key
        if not (0 <= x < self.orig_w and 0 <= y < self.orig_h):
            raise KeyError(key)                 # keeps the dict's bounds semantics
        if self.layout == "colmajor":
            return x * self.orig_h + y
        return (y // self.ps) * self.grid_cols + (x // self.ps)

    def __len__(self):
        return self.orig_w * self.orig_h        # every pixel is representable

    def __iter__(self):                          # only used if someone iterates
        for x in range(self.orig_w):
            for y in range(self.orig_h):
                yield (x, y)

    def coords(self) -> np.ndarray:
        """(K, 2) int array: row = embedding index, value = (x, y) in the
           ORIGINAL-PIXEL frame. This is what DistanceScoreGenerator needs,
           and it makes patch/pixel radii directly comparable."""
        if self.layout == "colmajor":
            k = np.arange(self.orig_w * self.orig_h)
            return np.stack([k // self.orig_h, k % self.orig_h], axis=1)
        k = np.arange(self.grid_rows * self.grid_cols)
        prow, pcol = k // self.grid_cols, k % self.grid_cols
        cx = pcol * self.ps + self.ps // 2       # patch-center pixel coords
        cy = prow * self.ps + self.ps // 2
        return np.stack([cx, cy], axis=1)