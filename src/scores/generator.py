import torch
import numpy as np
import time
import logging
from tqdm import tqdm
import gc
from scores.utils import get_landmark_indices, extract_landmark_embeddings, estimate_memory_gb

logger = logging.getLogger(__name__)


class ScoreGenerator:
    """
    Computes nonconformity scores from pre-generated embeddings.
 
    The score for a (source, target, candidate) triplet is:
        score = 1 - softmax(cosine_similarity / temperature)
 
    where the softmax is taken over all candidate locations in the target
    image, normalizing the cosine similarities into a probability distribution.
 
    Args:
        apply_softmax (bool): If True, apply softmax normalization over candidates
            before computing scores. If False, scores are raw 1 - cosine_similarity.
        temperature (float): Temperature parameter for softmax. Lower values make
            the distribution peakier (more confident).
        device (str): Device for GPU computation ('cuda', 'cpu', etc.).
        verbose (bool): If True, show progress bars and timing logs.
    """

    def __init__(self, apply_softmax: bool = True, temperature: float = 1.0, device: str = "cuda", verbose: bool = True ):
        self.apply_softmax = apply_softmax
        self.temperature = temperature
        self.device = device
        self.verbose = verbose
    
    def generate_calib_scores(self, embeddings: list[torch.Tensor], landmarks: list[list[float]], 
                              xy_maps: list[dict], return_all_scores: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Compute calibration nonconformity scores (leave-one-out).
 
        For each calibration image j, every other calibration source i
        scores against all K_j candidates in image j. The true-label score
        is extracted, and the self-score (i == j) is set to inf.
 
        Args:
            embeddings: List of N tensors, each (K_j, D).
            landmarks: List of N [x, y] coordinates (ground truth).
            xy_maps: List of N dicts mapping (x, y) -> index.
            return_all_scores: If True, also return the full (N, N, K_max)
                matrix of scores over all candidates (needed for model-selection
                methods). WARNING: this can be very large at pixel level.
 
        Returns:
            calib_true_matrix: (N, N) array of true-label nonconformity scores,
                with diagonal set to inf.
            calib_all_matrix (only if return_all_scores=True): (N, N, K_max)
                array of all-candidate scores, with diagonal slices set to inf.
        """
        start = time.perf_counter()
        N = len(embeddings)

        # Extract ground truth pixel number given the ground truth landmark
        gt_indices = get_landmark_indices(landmarks, xy_maps)

        # Extract the ground truth landmark embeddings
        lm_embeddings = extract_landmark_embeddings(landmarks, embeddings, xy_maps)

        with torch.no_grad():
            Q = torch.stack(lm_embeddings, dim=0).to(self.device)  # (N, D)
 
            if not self.apply_softmax:
                calib_true_matrix = self._calib_scores_no_softmax(Q, N)
                if return_all_scores:
                    raise NotImplementedError(
                        "return_all_scores requires apply_softmax=True. "
                        "Without softmax, there is no per-target normalization "
                        "over candidates."
                    )
            else:
                calib_true_matrix, calib_all_matrix = self._calib_scores_softmax(
                    Q, embeddings, gt_indices, N, return_all_scores
                )
 
        np.fill_diagonal(calib_true_matrix, np.inf)

        elapsed = time.perf_counter() - start
        logger.info(f"Calibration scores ({N}x{N}): {elapsed:.2f}s")
 
        if not return_all_scores:
            return calib_true_matrix
 
        return calib_true_matrix, calib_all_matrix
    
    def _calib_scores_no_softmax(self, Q: torch.Tensor, N: int) -> np.ndarray:
        """
        Fast path when softmax is disabled: scores are just 1 - cosine(landmark_i, landmark_j).
        Only computes landmark-to-landmark scores (no per-target candidate expansion).
        """
        cos_sims = Q @ Q.T  # (N, N)
        cos_sims.fill_diagonal_(float("-inf"))
        # Converting to numpy so that GPU memory don't exceed the limit
        return (1.0 - cos_sims).cpu().numpy()
    
    def _calib_scores_softmax(self, Q: torch.Tensor, embeddings: list[torch.Tensor], gt_indices: list[int], N: int,
        return_all_scores: bool) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Standard path: for each target j, compute softmax-normalized scores
        from all sources against all K_j candidates in target j.

        return_all_scores is required for modsel_cp, yk_baseline like methods
        SCOS and CAOS doesn't require this.
        """
        calib_true_scores = np.empty((N, N), dtype=np.float64)
        calib_all_list = [] if return_all_scores else None
 
        iterator = (
            tqdm(range(N), desc="Calibration scores", leave=False)
            if self.verbose
            else range(N)
        )
 
        for j in iterator:
            all_candidates_gpu = embeddings[j].to(self.device)  # (K_j, D)
            cos_sims = Q @ all_candidates_gpu.T  # (N, K_j)
 
            # Mask self-score before softmax so it doesn't affect normalization
            cos_sims[j, :] = torch.finfo(cos_sims.dtype).min
 
            cos_sims = torch.softmax(cos_sims / self.temperature, dim=1)
            scores = 1.0 - cos_sims  # (N, K_j)
 
            # Extract true-label score for target j
            calib_true_scores[:, j] = scores[:, gt_indices[j]].cpu().numpy()
 
            if return_all_scores:
                calib_all_list.append(scores.cpu().numpy())  # (N, K_j)
 
            del all_candidates_gpu, cos_sims, scores
 
        # Build all-label matrix if requested
        calib_all_matrix = None
        if return_all_scores:
            # Since images can be of different sizes
            K_max = max(arr.shape[1] for arr in calib_all_list)
            mem_gb = estimate_memory_gb((N, N, K_max))
            logger.info(
                f"Allocating calib_all_matrix: ({N}, {N}, {K_max}) = {mem_gb:.1f} GB"
            )
            calib_all_matrix = np.full((N, N, K_max), 1.1)
            for j, scores_j in enumerate(calib_all_list):
                K_j = scores_j.shape[1]
                calib_all_matrix[:, j, :K_j] = scores_j
            for i in range(N):
                calib_all_matrix[i, i, :] = np.inf
            del calib_all_list
            gc.collect()
 
        return calib_true_scores, calib_all_matrix
    
    def generate_eval_scores(self, test_embeddings: list[torch.Tensor], calib_lm_embeddings: list[torch.Tensor],) -> list[np.ndarray]:
        """
        Compute evaluation nonconformity scores for all test images.
 
        For each test image m, every calibration landmark embedding scores
        against all K_m candidate locations in the test image.
 
        Args:
            test_embeddings: List of M tensors, each (K_m, D).
            calib_lm_embeddings: List of N tensors, each (D,) — the landmark
                embeddings extracted from calibration images.
 
        Returns:
            List of M arrays, each (K_m, N). Entry [k, i] is the nonconformity
            score of candidate k in test image m according to calibration source i.
        """
        start = time.perf_counter()
        M = len(test_embeddings)
 
        Q_gpu = torch.stack(calib_lm_embeddings, dim=0).to(self.device)  # (N, D)
        all_scores = []
 
        iterator = (
            tqdm(range(M), desc="Evaluation scores", leave=False)
            if self.verbose
            else range(M)
        )
 
        with torch.no_grad():
            for m in iterator:
                test_gpu = test_embeddings[m].to(self.device)  # (K_m, D)
                cos_sims = Q_gpu @ test_gpu.T  # (N, K_m)
                del test_gpu
 
                if self.apply_softmax:
                    cos_sims = torch.softmax(cos_sims / self.temperature, dim=1)
 
                scores = 1.0 - cos_sims
                all_scores.append(scores.T.cpu().numpy())  # (K_m, N)
                del cos_sims
 
        elapsed = time.perf_counter() - start
        logger.info(
            f"Evaluation scores ({M} test images): {elapsed:.2f}s"
        )
 
        return all_scores