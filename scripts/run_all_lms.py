#!/usr/bin/env python3
"""
run_all_lms.py — Run the conformal landmark detection pipeline over ALL
landmarks in a dataset, reusing image embeddings across landmarks.

The dominant cost in run.py is the ViT forward pass for each image.  When
running L landmarks, calling run.py in a loop repeats that cost L times.
This script generates embeddings ONCE and reuses them for every landmark,
saving approximately (L-1)/L of the total embedding time.

Output CSV
----------
One row per (landmark, method), using the same column layout as run.py so
results from both scripts are directly comparable.  Results are flushed to
disk after each landmark so a crash mid-run loses at most one landmark.

Resume after a crash
--------------------
    python scripts/run_all_lms.py ... --start_landmark 12 -o results.csv

Usage
-----
    python scripts/run_all_lms.py \\
        -c data/calib.json -t data/test.json \\
        --base_img_path /data/images \\
        --level patch --alpha 0.1 --temperature 0.05 \\
        --methods caos scos -o results.csv
"""

import argparse
import csv
import gc
import json
import logging
import os
import time

import numpy as np
import torch

from oneshotlandmark.model import ViTModel
from oneshotlandmark.embeddings.patch import PatchEmbeddingGenerator
from oneshotlandmark.embeddings.pixel import PixelEmbeddingGenerator
from oneshotlandmark.embeddings.bilinear_interpolation import BilinearEmbeddingGenerator
from oneshotlandmark.scores.generator import ScoreGenerator
from oneshotlandmark.scores.utils import (
    estimate_memory_gb,
    extract_landmark_embeddings,
    get_landmark_indices,
    remove_self_2d,
    remove_self_3d,
    scores_to_matrix,
)
from cp4icl.oneshot import caos, scos
from cp4icl.model_selection import yk_baseline, yk_adjust, yk_split, modsel_cp, modsel_cp_ub

logger = logging.getLogger(__name__)

ONESHOT_METHODS = {"caos", "scos", "fullcaos"}
MODSEL_METHODS  = {"yk_baseline", "yk_adjust", "yk_split", "modsel_cp", "modsel_cp_ub"}
ALL_METHODS     = ONESHOT_METHODS | MODSEL_METHODS

# CSV columns written per (landmark × method) row.
_CSV_FIELDS = [
    "landmark_idx", "landmark_name",
    "level", "alpha", "temperature", "patch_size", "apply_softmax", "normalize", "k",
    "n_calib", "n_test",
    "method", "coverage", "avg_set_size",
    "landmark_time_seconds",
]


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_methods(methods_arg: list[str] | None) -> set[str]:
    if methods_arg is None:
        return {"caos", "scos"}
    if len(methods_arg) == 1 and methods_arg[0] == "all":
        return ALL_METHODS.copy()
    methods = set(methods_arg)
    unknown = methods - ALL_METHODS
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Valid: {sorted(ALL_METHODS)}")
    return methods


def _build_emb_gen(model, level: str, patch_size: int, normalize: bool, verbose: bool):
    """Factory: return the right embedding generator for the requested level."""
    if level == "patch":
        return PatchEmbeddingGenerator(
            model=model, patch_size=patch_size, normalize=normalize, verbose=verbose
        )
    if level == "pixel":
        return PixelEmbeddingGenerator(
            model=model, patch_size=patch_size, normalize=normalize, verbose=verbose
        )
    if level == "bilinear":
        return BilinearEmbeddingGenerator(
            model=model, patch_size=patch_size, normalize=normalize, verbose=verbose
        )
    raise ValueError(f"Unknown level '{level}'. Must be 'patch', 'pixel', or 'bilinear'.")


def _load_json(path: str):
    with open(path) as f:
        return json.load(f)


def _format_eta(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f} min"
    return f"{seconds/3600:.2f} hr"


# ─────────────────────────────────────────────────────────────────────────────
# Per-landmark runner  (embeddings are passed in; nothing is re-generated here)
# ─────────────────────────────────────────────────────────────────────────────

def run_single_landmark(
    *,
    lm_idx: int,
    calib_lms: list,
    test_lms: list,
    calib_embs: list[torch.Tensor],
    calib_xy_maps: list[dict],
    test_embs: list[torch.Tensor],
    test_xy_maps: list[dict],
    score_gen: ScoreGenerator,
    alpha: float,
    k: int,
    methods: set[str],
) -> dict:
    """
    Run all requested CP methods for a single landmark.

    Image embeddings are passed in pre-computed and are NOT modified.  All
    temporary tensors (score matrices, etc.) are allocated here and freed
    before the function returns.

    Args:
        lm_idx:        Landmark index (used only for log messages).
        calib_lms:     [x, y] ground-truth coords for each calibration image.
        test_lms:      [x, y] ground-truth coords for each test image.
        calib_embs:    Pre-computed (K_i, D) tensors, one per calibration image.
        calib_xy_maps: Pre-computed (x,y)->idx dicts, one per calibration image.
        test_embs:     Pre-computed (K_m, D) tensors, one per test image.
        test_xy_maps:  Pre-computed (x,y)->idx dicts, one per test image.
        score_gen:     Shared ScoreGenerator instance.
        alpha:         Miscoverage level.
        k:             Nearest sources for CAOS.
        methods:       Set of CP method names to run.

    Returns:
        Dict with n_calib, n_test, coverage_<method>, avg_set_size_<method>,
        and landmark_time_seconds.
    """
    needs_calib_all = bool(methods & MODSEL_METHODS)
    needs_no_self   = bool(methods & ({"scos"} | MODSEL_METHODS))

    t_start = time.perf_counter()

    N      = len(calib_lms)
    M      = len(test_lms)
    k_used = min(k, N)

    # ── Extract per-landmark calibration embeddings ──────────────────────────
    calib_lm_embs = extract_landmark_embeddings(calib_lms, calib_embs, calib_xy_maps)

    # ── Calibration scores ───────────────────────────────────────────────────
    logger.info(f"[lm {lm_idx}] Computing calibration scores (return_all={needs_calib_all})")
    if needs_calib_all:
        K_max_calib = max(e.shape[0] for e in calib_embs)
        mem_gb = estimate_memory_gb((N, N, K_max_calib))
        logger.info(
            f"[lm {lm_idx}] calib_all_matrix will be approx ({N}, {N}, {K_max_calib}) "
            f"= {mem_gb:.1f} GB"
        )
        calib_true_matrix, calib_all_matrix = score_gen.generate_calib_scores(
            calib_embs, calib_lms, calib_xy_maps, return_all_scores=True
        )
    else:
        calib_true_matrix = score_gen.generate_calib_scores(
            calib_embs, calib_lms, calib_xy_maps, return_all_scores=False
        )
        calib_all_matrix = None

    # ── Eval scores ──────────────────────────────────────────────────────────
    logger.info(f"[lm {lm_idx}] Computing eval scores")
    eval_scores_list   = score_gen.generate_eval_scores(test_embs, calib_lm_embs)
    eval_scores_matrix = scores_to_matrix(eval_scores_list)   # (N, M, K_max)
    K_eval             = eval_scores_matrix.shape[2]
    del eval_scores_list
    gc.collect()

    # ── Test labels ──────────────────────────────────────────────────────────
    test_labels = np.array(get_landmark_indices(test_lms, test_xy_maps))

    # ── Self-removed calibration matrices ────────────────────────────────────
    if needs_no_self:
        calib_true_no_self = remove_self_2d(calib_true_matrix)       # (N, N-1)

    if needs_calib_all:
        # Harmonize K dimension between calib and eval matrices when image sizes differ.
        K_calib = calib_all_matrix.shape[2]
        if K_calib != K_eval:
            K_unified = max(K_calib, K_eval)
            logger.info(
                f"[lm {lm_idx}] K mismatch — harmonizing: "
                f"calib={K_calib}, eval={K_eval} → {K_unified}"
            )
            if K_calib < K_unified:
                calib_all_matrix = np.pad(
                    calib_all_matrix,
                    ((0, 0), (0, 0), (0, K_unified - K_calib)),
                    constant_values=1.1,
                )
            if K_eval < K_unified:
                eval_scores_matrix = np.pad(
                    eval_scores_matrix,
                    ((0, 0), (0, 0), (0, K_unified - K_eval)),
                    constant_values=1.1,
                )

        calib_all_no_self = remove_self_3d(calib_all_matrix)         # (N, N-1, K)
        del calib_all_matrix
        gc.collect()

    # ── CP methods ───────────────────────────────────────────────────────────
    results = {"n_calib": N, "n_test": M}

    if "caos" in methods:
        r = caos(
            alpha=alpha,
            calib_scores=calib_true_matrix,
            eval_scores=eval_scores_matrix,
            y_eval=test_labels,
            k=k_used,
        )
        results["coverage_caos"]      = r.coverage
        results["avg_set_size_caos"]  = r.avg_set_size
        logger.info(f"[lm {lm_idx}] CAOS      coverage={r.coverage:.4f}  avg_set_size={r.avg_set_size:.2f}")

    if "scos" in methods:
        r = scos(
            alpha=alpha,
            calib_scores=calib_true_no_self,
            eval_scores=eval_scores_matrix,
            y_eval=test_labels,
        )
        results["coverage_scos"]      = r.coverage
        results["avg_set_size_scos"]  = r.avg_set_size
        logger.info(f"[lm {lm_idx}] SCOS      coverage={r.coverage:.4f}  avg_set_size={r.avg_set_size:.2f}")

    if "fullcaos" in methods:
        # Reverse scores are not yet implemented; skip rather than crash.
        logger.warning(
            f"[lm {lm_idx}] fullcaos skipped — reverse score computation "
            "is not yet implemented (see run.py Phase 4)."
        )

    if "yk_baseline" in methods:
        r = yk_baseline(
            alpha=alpha,
            calib_true_scores=calib_true_no_self,
            calib_all_scores=calib_all_no_self,
            eval_scores=eval_scores_matrix,
            y_eval=test_labels,
        )
        results["coverage_yk_baseline"]     = r.coverage
        results["avg_set_size_yk_baseline"] = r.avg_set_size
        logger.info(f"[lm {lm_idx}] yk_baseline  coverage={r.coverage:.4f}  avg_set_size={r.avg_set_size:.2f}")

    if "yk_adjust" in methods:
        r = yk_adjust(
            alpha=alpha,
            calib_true_scores=calib_true_no_self,
            calib_all_scores=calib_all_no_self,
            eval_scores=eval_scores_matrix,
            y_eval=test_labels,
        )
        results["coverage_yk_adjust"]     = r.coverage
        results["avg_set_size_yk_adjust"] = r.avg_set_size
        logger.info(f"[lm {lm_idx}] yk_adjust    coverage={r.coverage:.4f}  avg_set_size={r.avg_set_size:.2f}")

    if "yk_split" in methods:
        r = yk_split(
            alpha=alpha,
            calib_true_scores=calib_true_no_self,
            calib_all_scores=calib_all_no_self,
            eval_scores=eval_scores_matrix,
            y_eval=test_labels,
            shuffle=True,
            random_state=42,
        )
        results["coverage_yk_split"]     = r.coverage
        results["avg_set_size_yk_split"] = r.avg_set_size
        logger.info(f"[lm {lm_idx}] yk_split     coverage={r.coverage:.4f}  avg_set_size={r.avg_set_size:.2f}")

    if "modsel_cp" in methods:
        r = modsel_cp(
            alpha=alpha,
            calib_true_scores=calib_true_no_self,
            calib_all_scores=calib_all_no_self,
            eval_scores=eval_scores_matrix,
            y_eval=test_labels,
        )
        results["coverage_modsel_cp"]     = r.coverage
        results["avg_set_size_modsel_cp"] = r.avg_set_size
        logger.info(f"[lm {lm_idx}] modsel_cp    coverage={r.coverage:.4f}  avg_set_size={r.avg_set_size:.2f}")

    if "modsel_cp_ub" in methods:
        r = modsel_cp_ub(
            alpha=alpha,
            calib_true_scores=calib_true_no_self,
            calib_all_scores=calib_all_no_self,
            eval_scores=eval_scores_matrix,
            y_eval=test_labels,
        )
        results["coverage_modsel_cp_ub"]     = r.coverage
        results["avg_set_size_modsel_cp_ub"] = r.avg_set_size
        logger.info(f"[lm {lm_idx}] modsel_cp_ub coverage={r.coverage:.4f}  avg_set_size={r.avg_set_size:.2f}")

    # ── Per-landmark cleanup ─────────────────────────────────────────────────
    del calib_true_matrix, eval_scores_matrix
    if needs_no_self:
        del calib_true_no_self
    if needs_calib_all:
        del calib_all_no_self
    gc.collect()
    torch.cuda.empty_cache()

    results["landmark_time_seconds"] = time.perf_counter() - t_start
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run conformal landmark detection over ALL landmarks in a dataset. "
            "Image embeddings are generated once and shared across all landmarks."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ─────────────────────────────────────────────────────────────────
    parser.add_argument("-c", "--calib_path", required=True,
                        help="JSON file for calibration images")
    parser.add_argument("-t", "--test_path", required=True,
                        help="JSON file for test images")
    parser.add_argument("--base_img_path", required=True,
                        help="Base directory prepended to image paths in JSON")
    parser.add_argument("--landmark_names_path", default=None,
                        help="Optional JSON file with a list of landmark name strings. "
                             "If omitted, landmarks are named landmark_0, landmark_1, …")

    # ── Embedding ────────────────────────────────────────────────────────────
    parser.add_argument("--level", choices=["patch", "pixel", "bilinear"], default="patch",
                        help="Embedding granularity")
    parser.add_argument("--patch_size", type=int, default=16,
                        help="ViT patch size in pixels")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True,
                        help="Mean-centre embeddings before L2 normalisation")

    # ── Scoring ──────────────────────────────────────────────────────────────
    parser.add_argument("--temperature", type=float, default=0.05,
                        help="Softmax temperature for nonconformity scores")
    parser.add_argument("--softmax", action=argparse.BooleanOptionalAction, default=True,
                        help="Apply softmax normalisation to scores")

    # ── Conformal prediction ─────────────────────────────────────────────────
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Miscoverage level (e.g. 0.1 → 90%% coverage target)")
    parser.add_argument("--k", type=int, default=3,
                        help="Number of nearest sources for CAOS")
    parser.add_argument("--methods", nargs="+", default=None,
                        help='CP methods to run, or "all". '
                             'Choices: caos scos fullcaos yk_baseline yk_adjust '
                             'yk_split modsel_cp modsel_cp_ub')

    # ── Resume ───────────────────────────────────────────────────────────────
    parser.add_argument("--start_landmark", type=int, default=0,
                        help="Start (or resume) from this landmark index. "
                             "When > 0, results are appended to --output_path.")

    # ── Infrastructure ───────────────────────────────────────────────────────
    parser.add_argument("--device", default="cuda", help="Torch device")
    parser.add_argument("-o", "--output_path", default=None,
                        help="CSV file for results (one row per landmark × method)")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Enable DEBUG-level logging")

    args = parser.parse_args()

    # ── Logging ──────────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    methods = _parse_methods(args.methods)

    # ── Load JSON data ────────────────────────────────────────────────────────
    calib_json = _load_json(args.calib_path)
    test_json  = _load_json(args.test_path)

    calib_img_paths = [os.path.join(args.base_img_path, e["img_path"]) for e in calib_json]
    test_img_paths  = [os.path.join(args.base_img_path, e["img_path"]) for e in test_json]

    n_landmarks = len(calib_json[0]["landmarks"])

    if args.landmark_names_path:
        landmark_names = _load_json(args.landmark_names_path)
        if len(landmark_names) != n_landmarks:
            raise ValueError(
                f"landmark_names_path has {len(landmark_names)} entries "
                f"but data has {n_landmarks} landmarks."
            )
    else:
        landmark_names = [f"landmark_{i}" for i in range(n_landmarks)]

    logger.info(
        f"Dataset: {len(calib_img_paths)} calib, {len(test_img_paths)} test, "
        f"{n_landmarks} landmarks"
    )
    logger.info(f"Level={args.level}  methods={sorted(methods)}")
    logger.info(f"alpha={args.alpha}  temperature={args.temperature}  k={args.k}")

    # ── Model + generators (shared across all landmarks) ─────────────────────
    logger.info("Initialising ViT model")
    model     = ViTModel(device_str=args.device)
    emb_gen   = _build_emb_gen(model, args.level, args.patch_size, args.normalize, verbose=True)
    score_gen = ScoreGenerator(
        apply_softmax=args.softmax,
        temperature=args.temperature,
        device=args.device,
    )

    # ── Generate embeddings ONCE ──────────────────────────────────────────────
    t_total_start = time.perf_counter()

    logger.info("Generating calibration embeddings (shared across all landmarks) …")
    calib_embs, calib_xy_maps = emb_gen.generate_embedding_all(calib_img_paths)

    logger.info("Generating test embeddings (shared across all landmarks) …")
    test_embs, test_xy_maps = emb_gen.generate_embedding_all(test_img_paths)

    t_emb = time.perf_counter() - t_total_start
    logger.info(f"Embedding generation done: {_format_eta(t_emb)}")

    # ── CSV setup ─────────────────────────────────────────────────────────────
    # Append when resuming (start_landmark > 0); overwrite when starting fresh.
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        csv_mode    = "a" if args.start_landmark > 0 else "w"
        write_header = csv_mode == "w"
        csv_file     = open(args.output_path, csv_mode, newline="")
        writer       = csv.DictWriter(csv_file, fieldnames=_CSV_FIELDS)
        if write_header:
            writer.writeheader()
    else:
        csv_file = writer = None

    # ── Per-landmark loop ─────────────────────────────────────────────────────
    landmark_times: list[float] = []

    for lm_idx in range(args.start_landmark, n_landmarks):
        lm_name   = landmark_names[lm_idx]
        calib_lms = [e["landmarks"][lm_idx] for e in calib_json]
        test_lms  = [e["landmarks"][lm_idx] for e in test_json]

        print(f"\n{'='*60}")
        print(f"  Landmark {lm_idx + 1}/{n_landmarks}  —  {lm_name}")
        print(f"{'='*60}")

        try:
            lm_results = run_single_landmark(
                lm_idx=lm_idx,
                calib_lms=calib_lms,
                test_lms=test_lms,
                calib_embs=calib_embs,
                calib_xy_maps=calib_xy_maps,
                test_embs=test_embs,
                test_xy_maps=test_xy_maps,
                score_gen=score_gen,
                alpha=args.alpha,
                k=args.k,
                methods=methods,
            )

            lm_time = lm_results["landmark_time_seconds"]
            landmark_times.append(lm_time)

            # Per-landmark results summary
            for method in sorted(methods):
                cov = lm_results.get(f"coverage_{method}")
                sz  = lm_results.get(f"avg_set_size_{method}")
                if cov is not None:
                    print(f"  {method:<16} coverage={cov:.4f}  avg_set_size={sz:.2f}")

            # ETA
            avg_t     = sum(landmark_times) / len(landmark_times)
            remaining = n_landmarks - lm_idx - 1
            print(
                f"\n  Landmark time: {_format_eta(lm_time)}  |  "
                f"Avg: {_format_eta(avg_t)}/lm  |  "
                f"ETA: {_format_eta(remaining * avg_t)}"
            )

            # Write CSV rows (one per method) and flush immediately
            if writer:
                base = {
                    "landmark_idx":          lm_idx,
                    "landmark_name":         lm_name,
                    "level":                 args.level,
                    "alpha":                 args.alpha,
                    "temperature":           args.temperature,
                    "patch_size":            args.patch_size,
                    "apply_softmax":         args.softmax,
                    "normalize":             args.normalize,
                    "k":                     args.k,
                    "n_calib":               lm_results["n_calib"],
                    "n_test":                lm_results["n_test"],
                    "landmark_time_seconds": lm_time,
                }
                for method in sorted(methods):
                    writer.writerow({
                        **base,
                        "method":       method,
                        "coverage":     lm_results.get(f"coverage_{method}",     ""),
                        "avg_set_size": lm_results.get(f"avg_set_size_{method}", ""),
                    })
                csv_file.flush()

        except Exception as exc:
            logger.exception(f"Landmark {lm_idx} ({lm_name}) FAILED: {exc}")
            if writer:
                base = {
                    "landmark_idx": lm_idx, "landmark_name": lm_name,
                    "level": args.level, "alpha": args.alpha,
                    "temperature": args.temperature, "patch_size": args.patch_size,
                    "apply_softmax": args.softmax, "normalize": args.normalize,
                    "k": args.k, "n_calib": "", "n_test": "",
                    "landmark_time_seconds": "",
                }
                for method in sorted(methods):
                    writer.writerow({**base, "method": method,
                                     "coverage": "ERROR", "avg_set_size": "ERROR"})
                csv_file.flush()

    # ── Final summary ─────────────────────────────────────────────────────────
    if csv_file:
        csv_file.close()

    total_time     = time.perf_counter() - t_total_start
    n_succeeded    = len(landmark_times)
    n_ran          = n_landmarks - args.start_landmark

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"  Landmarks:        {n_succeeded}/{n_ran} succeeded")
    print(f"  Embedding time:   {_format_eta(t_emb)}")
    if landmark_times:
        print(f"  Mean per-lm:      {_format_eta(sum(landmark_times)/len(landmark_times))}")
    print(f"  Total time:       {_format_eta(total_time)}")
    if args.output_path:
        print(f"  Results saved to: {args.output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
