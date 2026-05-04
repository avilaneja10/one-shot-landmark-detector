from oneshotlandmark.model import ViTModel
from oneshotlandmark.embeddings.patch import PatchEmbeddingGenerator
from oneshotlandmark.embeddings.pixel import PixelEmbeddingGenerator
from oneshotlandmark.embeddings.bilinear_interpolation import BilinearEmbeddingGenerator
from oneshotlandmark.scores.generator import ScoreGenerator
from oneshotlandmark.scores.utils import extract_landmark_embeddings, scores_to_matrix, get_landmark_indices, remove_self_2d
from cp4icl.oneshot import caos, scos, fullcaos
from cp4icl.model_selection import yk_baseline, yk_adjust, yk_split, modsel_cp, modsel_cp_ub
import json
import logging
import os
import time
import gc
import torch
import numpy as np
import argparse
import csv

logger  = logging.getLogger(__name__)

# REGISTER METHODS
ONESHOT_METHODS = {"caos", "scos", "fullcaos"}
MODSEL_METHODS = {"yk_baseline", "yk_adjust", "yk_split", "modsel_cp", "modsel_cp_ub"}
ALL_METHODS = ONESHOT_METHODS | MODSEL_METHODS

def parse_methods(methods_arg):
    """Parse and validate the methods argument."""
    if methods_arg is None:
        return {"caos", "scos"}
    if len(methods_arg) == 1 and methods_arg[0] == "all":
        return ALL_METHODS.copy()
    methods = set(methods_arg)
    unknown = methods - ALL_METHODS
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Valid: {sorted(ALL_METHODS)}")
    return methods

def load_data(calib_path, test_path, base_img_path, landmark_idx):
    """
    Load calibration and test JSON files. Extract image paths and landmarks.
 
    Expected JSON format: list of dicts, each with:
        - "img_path": relative path to the image
        - "landmarks": list of [x, y] coordinates
    """
    with open(calib_path) as f:
        calib_json = json.load(f)
    with open(test_path) as f:
        test_json = json.load(f)
 
    calib_img_paths = [os.path.join(base_img_path, x["img_path"]) for x in calib_json]
    test_img_paths = [os.path.join(base_img_path, x["img_path"]) for x in test_json]
 
    calib_lms = [entry["landmarks"][landmark_idx] for entry in calib_json]
    test_lms = [entry["landmarks"][landmark_idx] for entry in test_json]
 
    logger.info(
        f"Loaded data: {len(calib_img_paths)} calib images, "
        f"{len(test_img_paths)} test images, landmark_idx={landmark_idx}"
    )
 
    return calib_img_paths, test_img_paths, calib_lms, test_lms

def run(calib_img_paths, test_img_paths, calib_lms, test_lms, level="patch",
    alpha=0.1, temperature=0.05, patch_size=16, apply_softmax=True,
    normalize=True, k=3, device="cuda", methods=None):
    """
    Run conformal prediction pipeline.
 
    Args:
        calib_img_paths: List of calibration image file paths.
        test_img_paths: List of test image file paths.
        calib_lms: List of [x, y] landmark coordinates for calibration images.
        test_lms: List of [x, y] landmark coordinates for test images.
        level: Embedding granularity — "patch" or "pixel".
        alpha: Miscoverage level (e.g., 0.1 for 90% coverage target).
        temperature: Softmax temperature for nonconformity scores.
        patch_size: ViT patch size in pixels.
        apply_softmax: Whether to apply softmax normalization.
        normalize: Whether to mean-center embeddings before L2 normalization.
        k: Number of nearest sources to average for CAOS/fullCAOS.
        device: Torch device string.
        methods: Set of CP method names to run.
 
    Returns:
        Dict with coverage and avg_set_size for each method.
    """
    if methods is None:
        methods = {"caos", "scos"}
 
    needs_calib_all = bool(methods & MODSEL_METHODS)
    needs_reverse = "fullcaos" in methods
 
    total_start = time.perf_counter()

    # ==================================================================
    # Phase 1: Model + Embedding Generator
    # ==================================================================
    logger.info(f"Initializing model and {level}-level embedding generator")

    model = ViTModel(device_str=device)

    if level == "patch":
        emb_gen = PatchEmbeddingGenerator(
            model=model, patch_size=patch_size, normalize=normalize
        )
    elif level == "pixel":
        emb_gen = PixelEmbeddingGenerator(
            model=model, patch_size=patch_size, normalize=normalize
        )
    elif level == "bilinear":
        emb_gen = BilinearEmbeddingGenerator(
            model=model, patch_size=patch_size, normalize=normalize
        )
    else:
        raise ValueError(f"Unknown level: {level}. Must be 'patch', 'pixel', or 'bilinear'.")
    
    score_gen = ScoreGenerator(
        apply_softmax=apply_softmax,
        temperature=temperature,
        device=device,
    )

    # ==================================================================
    # Phase 2: Calibration embeddings + scores
    # ==================================================================
    logger.info("Generating calibration embeddings")
    calib_embs, calib_xy_maps = emb_gen.generate_embedding_all(calib_img_paths)
 
    calib_lm_embs = extract_landmark_embeddings(calib_lms, calib_embs, calib_xy_maps)
 
    logger.info("Computing calibration scores")
    if needs_calib_all:
        calib_true_matrix, calib_all_matrix = score_gen.generate_calib_scores(
            calib_embs, calib_lms, calib_xy_maps, return_all_scores=True
        )
    else:
        calib_true_matrix = score_gen.generate_calib_scores(
            calib_embs, calib_lms, calib_xy_maps, return_all_scores=False
        )
    

    # Free calib embeddings unless needed for reverse scores
    if not needs_reverse:
        del calib_embs, calib_xy_maps
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Freed calibration embeddings")

    # ==================================================================
    # Phase 3: Test embeddings + evaluation scores
    # ==================================================================
    logger.info("Generating test embeddings")
    test_embs, test_xy_maps = emb_gen.generate_embedding_all(test_img_paths)
 
    logger.info("Computing evaluation scores")
    eval_scores_list = score_gen.generate_eval_scores(test_embs, calib_lm_embs)
    eval_scores_matrix = scores_to_matrix(eval_scores_list)
    K_max = eval_scores_matrix.shape[2]
 
    del eval_scores_list
    gc.collect()

    # ==================================================================
    # Phase 4: Reverse scores (fullCAOS only)
    # CURRENTLY NOT IMPLEMENTED 
    # ==================================================================
    # reverse_matrix = None
    # if needs_reverse:
    #     logger.info("Computing reverse scores for fullCAOS")
    #     reverse_matrix = score_gen.generate_reverse_scores(
    #         test_embs, calib_embs, calib_lms, calib_xy_maps, K_max=K_max
    #     )
    #     del calib_embs, calib_xy_maps
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #     logger.info("Freed calibration embeddings after reverse scores")
 
    # Free test embeddings
    del test_embs
    gc.collect()


    # ==================================================================
    # Phase 5: Test labels
    # ==================================================================
    test_labels = np.array(get_landmark_indices(test_lms, test_xy_maps))
    del test_xy_maps
 
    k = min(k, len(calib_img_paths))

    # ==================================================================
    # Phase 6: Prepare self-removed matrices
    # ==================================================================
    needs_no_self = bool(methods & ({"scos"} | MODSEL_METHODS))
 
    if needs_no_self:
        calib_true_no_self = remove_self_2d(calib_true_matrix)  # (N, N-1)
 
    if needs_calib_all:
        # Harmonize K: calib and eval images may have different candidate counts
        K_calib = calib_all_matrix.shape[2]
        K_eval = eval_scores_matrix.shape[2]
        if K_calib != K_eval:
            K_unified = max(K_calib, K_eval)
            logger.info(f"Harmonizing K dimensions: calib={K_calib}, eval={K_eval} -> {K_unified}")
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
 
        calib_all_no_self = remove_self_3d(calib_all_matrix)  # (N, N-1, K)
        del calib_all_matrix
        gc.collect()
 
    # ==================================================================
    # Phase 7: Run CP methods
    # ==================================================================
    logger.info(f"Running methods: {sorted(methods)}")
 
    results = {
        "n_calib": len(calib_img_paths),
        "n_test": len(test_img_paths),
    }
 
    if "caos" in methods:
        r = caos(
            alpha=alpha, calib_scores=calib_true_matrix,
            eval_scores=eval_scores_matrix, y_eval=test_labels, k=k,
        )
        results["coverage_caos"] = r.coverage
        results["avg_set_size_caos"] = r.avg_set_size
        logger.info(f"CAOS: coverage={r.coverage:.4f}, avg_set_size={r.avg_set_size:.2f}")
 
    if "scos" in methods:
        r = scos(
            alpha=alpha, calib_scores=calib_true_no_self,
            eval_scores=eval_scores_matrix, y_eval=test_labels,
        )
        results["coverage_scos"] = r.coverage
        results["avg_set_size_scos"] = r.avg_set_size
        logger.info(f"SCOS: coverage={r.coverage:.4f}, avg_set_size={r.avg_set_size:.2f}")
 
    if "fullcaos" in methods:
        r = fullcaos(
            alpha=alpha, calib_scores=calib_true_matrix,
            eval_scores=eval_scores_matrix,
            reverse_scores=reverse_matrix,
            y_eval=test_labels, k=k,
        )
        results["coverage_fullcaos"] = r.coverage
        results["avg_set_size_fullcaos"] = r.avg_set_size
        logger.info(f"fullCAOS: coverage={r.coverage:.4f}, avg_set_size={r.avg_set_size:.2f}")
 
    if "yk_baseline" in methods:
        r = yk_baseline(
            alpha=alpha, calib_true_scores=calib_true_no_self,
            calib_all_scores=calib_all_no_self,
            eval_scores=eval_scores_matrix, y_eval=test_labels,
        )
        results["coverage_yk_baseline"] = r.coverage
        results["avg_set_size_yk_baseline"] = r.avg_set_size
        logger.info(f"yk_baseline: coverage={r.coverage:.4f}, avg_set_size={r.avg_set_size:.2f}")
 
    if "yk_adjust" in methods:
        r = yk_adjust(
            alpha=alpha, calib_true_scores=calib_true_no_self,
            calib_all_scores=calib_all_no_self,
            eval_scores=eval_scores_matrix, y_eval=test_labels,
        )
        results["coverage_yk_adjust"] = r.coverage
        results["avg_set_size_yk_adjust"] = r.avg_set_size
        logger.info(f"yk_adjust: coverage={r.coverage:.4f}, avg_set_size={r.avg_set_size:.2f}")
 
    if "yk_split" in methods:
        r = yk_split(
            alpha=alpha, calib_true_scores=calib_true_no_self,
            calib_all_scores=calib_all_no_self,
            eval_scores=eval_scores_matrix, y_eval=test_labels,
            shuffle=True, random_state=42,
        )
        results["coverage_yk_split"] = r.coverage
        results["avg_set_size_yk_split"] = r.avg_set_size
        logger.info(f"yk_split: coverage={r.coverage:.4f}, avg_set_size={r.avg_set_size:.2f}")
 
    if "modsel_cp" in methods:
        r = modsel_cp(
            alpha=alpha, calib_true_scores=calib_true_no_self,
            calib_all_scores=calib_all_no_self,
            eval_scores=eval_scores_matrix, y_eval=test_labels,
        )
        results["coverage_modsel_cp"] = r.coverage
        results["avg_set_size_modsel_cp"] = r.avg_set_size
        logger.info(f"modsel_cp: coverage={r.coverage:.4f}, avg_set_size={r.avg_set_size:.2f}")
 
    if "modsel_cp_ub" in methods:
        r = modsel_cp_ub(
            alpha=alpha, calib_true_scores=calib_true_no_self,
            calib_all_scores=calib_all_no_self,
            eval_scores=eval_scores_matrix, y_eval=test_labels,
        )
        results["coverage_modsel_cp_ub"] = r.coverage
        results["avg_set_size_modsel_cp_ub"] = r.avg_set_size
        logger.info(f"modsel_cp_ub: coverage={r.coverage:.4f}, avg_set_size={r.avg_set_size:.2f}")
 
    total_elapsed = time.perf_counter() - total_start
    results["total_time_seconds"] = total_elapsed
    logger.info(f"Total runtime: {total_elapsed:.2f}s")
 
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Conformal prediction for one-shot landmark localization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
 
    # Required
    parser.add_argument("-c", "--calib_path", required=True,
                        help="JSON path for calibration images")
    parser.add_argument("-t", "--test_path", required=True,
                        help="JSON path for test images")
    parser.add_argument("-l", "--landmark_idx", type=int, required=True,
                        help="Index of the landmark to evaluate")
 
    # Embedding level
    parser.add_argument("--level", choices=["patch", "pixel", "bilinear"], default="patch",
                        help="Embedding granularity")
 
    # Score computation
    parser.add_argument("--temperature", type=float, default=0.05,
                        help="Softmax temperature")
    parser.add_argument("--softmax", action=argparse.BooleanOptionalAction, default=True,
                        help="Apply softmax normalization")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True,
                        help="Mean-center embeddings before L2 normalization")
 
    # Conformal prediction
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Miscoverage level (e.g., 0.1 for 90%% target coverage)")
    parser.add_argument("--k", type=int, default=3,
                        help="Number of nearest sources for CAOS/fullCAOS")
    parser.add_argument("--methods", nargs="+", default=None,
                        help='CP methods to run (caos scos fullcaos yk_baseline '
                             'yk_adjust yk_split modsel_cp modsel_cp_ub), or "all"')
 
    # Paths and infra
    parser.add_argument("--base_img_path", required=True,
                        help="Base directory prepended to image paths in JSON")
    parser.add_argument("--patch_size", type=int, default=16,
                        help="ViT patch size in pixels")
    parser.add_argument("--device", default="cuda",
                        help="Torch device")
    parser.add_argument("-o", "--output_path", default=None,
                        help="Path to save results as csv")
 
    # Logging
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Enable DEBUG-level logging")
 
    args = parser.parse_args()
 
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
 
    # Load data
    calib_img_paths, test_img_paths, calib_lms, test_lms = load_data(
        args.calib_path, args.test_path, args.base_img_path, args.landmark_idx
    )
 
    # Parse methods
    methods = parse_methods(args.methods)
 
    # Run
    results = run(
        calib_img_paths=calib_img_paths,
        test_img_paths=test_img_paths,
        calib_lms=calib_lms,
        test_lms=test_lms,
        level=args.level,
        alpha=args.alpha,
        temperature=args.temperature,
        patch_size=args.patch_size,
        apply_softmax=args.softmax,
        normalize=args.normalize,
        k=args.k,
        device=args.device,
        methods=methods,
    )
 
    # Print results
    print("\n" + "=" * 60)
    print(f"Results  (level={args.level}, alpha={args.alpha}, temp={args.temperature})")
    print("=" * 60)
    for key, val in results.items():
        if key.startswith("coverage_"):
            method = key.replace("coverage_", "")
            size_val = results.get(f"avg_set_size_{method}", "N/A")
            print(f"  {method:<15} coverage={val:.4f}  avg_set_size={size_val:.2f}")
    print(f"\n  Total time: {results['total_time_seconds']:.2f}s")
    print("=" * 60)
 
    # Save results as CSV — one row per method, appends if file exists.
    # Fixed columns make it safe to append runs with different methods.
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
 
        fieldnames = [
            "level", "landmark_idx", "alpha", "temperature", "patch_size",
            "apply_softmax", "normalize", "k", "n_calib", "n_test",
            "method", "coverage", "avg_set_size", "total_time_seconds",
        ]
 
        base_row = {
            "level": args.level,
            "landmark_idx": args.landmark_idx,
            "alpha": args.alpha,
            "temperature": args.temperature,
            "patch_size": args.patch_size,
            "apply_softmax": args.softmax,
            "normalize": args.normalize,
            "k": args.k,
            "n_calib": results["n_calib"],
            "n_test": results["n_test"],
            "total_time_seconds": results["total_time_seconds"],
        }
 
        needs_header = not os.path.exists(args.output_path) or os.path.getsize(args.output_path) == 0
        with open(args.output_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if needs_header:
                writer.writeheader()
            for method in sorted(methods):
                row = {
                    **base_row,
                    "method": method,
                    "coverage": results.get(f"coverage_{method}", ""),
                    "avg_set_size": results.get(f"avg_set_size_{method}", ""),
                }
                writer.writerow(row)
 
        logger.info(f"Results saved to {args.output_path}") 
 
if __name__ == "__main__":
    main()