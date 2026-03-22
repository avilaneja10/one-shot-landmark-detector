import torch
import numpy as np
import time
import logging
from tqdm import tqdm
import gc
from oneshotlandmark.scores.utils import get_landmark_indices, extract_landmark_embeddings, estimate_memory_gb

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

    def __init__(self, level: str = "patch", patch_size: int = 16, apply_softmax: bool = True, 
                 temperature: float = 1.0, device: str = "cuda", verbose: bool = True ):
        self.apply_softmax = apply_softmax
        self.temperature = temperature
        self.device = device
        self.verbose = verbose
        self.level = level
        self.patch_size = patch_size

    def compute_calib_cosines(self, embeddings: list[torch.Tensor], landmarks: list[list[float]], image_dims) -> dict:
        """
        Compute raw cosine similarities for calibration (leave-one-out).
 
        For each calibration target j, computes the cosine similarity between
        every source landmark embedding and all K_j candidates in target j.
 
        The result is temperature-independent and can be cached.
 
        Args:
            embeddings: List of N tensors, each (K_j, D).
            landmarks: List of N [x, y] ground-truth coordinates.
 
        Returns:
            Dict with:
                "cosines": list of N numpy arrays, each (N, K_j) float32.
                    Raw cosine similarities. Self-scores are NOT masked.
                "gt_indices": list of N ints — true-label index per target.
                "lm_embeddings": list of N (D,) tensors — landmark embeddings.
        """
        start = time.perf_counter()
        N = len(embeddings)
 
        gt_indices = get_landmark_indices(landmarks, image_dims, self.level, self.patch_size)
        lm_embeddings = extract_landmark_embeddings(landmarks, embeddings, image_dims, self.level, self.patch_size)
 
        Q_gpu = torch.stack(lm_embeddings, dim=0).to(self.device)  # (N, D)
 
        cosines_list = []
 
        iterator = (
            tqdm(range(N), desc="Calib cosines", leave=False)
            if self.verbose else range(N)
        )
 
        with torch.no_grad():
            for j in iterator:
                candidates_gpu = embeddings[j].to(self.device)  # (K_j, D)
                cos = Q_gpu @ candidates_gpu.T  # (N, K_j)
                cosines_list.append(cos.cpu().numpy().astype(np.float32))
                del candidates_gpu
 
        elapsed = time.perf_counter() - start
        logger.info(f"Calib cosines ({N} targets): {elapsed:.2f}s")
 
        return {
            "cosines": cosines_list,
            "gt_indices": gt_indices,
            "lm_embeddings": lm_embeddings,
        }
        
    def compute_eval_cosines(
        self,
        test_embeddings: list[torch.Tensor],
        calib_lm_embeddings: list[torch.Tensor],
    ) -> dict:
        """
        Compute raw cosine similarities for evaluation.
 
        For each test image m, computes the cosine similarity between every
        calibration landmark embedding and all K_m candidates in the test image.
 
        The result is temperature-independent and can be cached.
 
        Args:
            test_embeddings: List of M tensors, each (K_m, D).
            calib_lm_embeddings: List of N tensors, each (D,).
 
        Returns:
            Dict with:
                "cosines": list of M numpy arrays, each (N, K_m) float32.
        """
        start = time.perf_counter()
        M = len(test_embeddings)
 
        Q_gpu = torch.stack(calib_lm_embeddings, dim=0).to(self.device)  # (N, D)
 
        cosines_list = []
 
        iterator = (
            tqdm(range(M), desc="Eval cosines", leave=False)
            if self.verbose else range(M)
        )
 
        with torch.no_grad():
            for m in iterator:
                test_gpu = test_embeddings[m].to(self.device)  # (K_m, D)
                cos = Q_gpu @ test_gpu.T  # (N, K_m)
                cosines_list.append(cos.cpu().numpy().astype(np.float32))
                del test_gpu
 
        elapsed = time.perf_counter() - start
        logger.info(f"Eval cosines ({M} test images): {elapsed:.2f}s")
 
        return {"cosines": cosines_list}

    # =====================================================================
    # We apply temperature and softmax now in the following functions
    # This is done to make sure that we are able to cache raw cosine scores
    # ======================================================================
 
    @staticmethod
    def apply_calib_scoring(
        calib_cosines: dict,
        temperature: float = 1.0,
        apply_softmax: bool = True,
        return_all_scores: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Convert raw calibration cosines into nonconformity scores.
 
        Applies self-masking, optional temperature scaling + softmax, and
        extracts true-label scores.
 
        Args:
            calib_cosines: Dict from compute_calib_cosines.
            temperature: Softmax temperature.
            apply_softmax: Whether to apply softmax normalization.
            return_all_scores: If True, also return (N, N, K_max) all-candidate scores.
 
        Returns:
            calib_true_matrix: (N, N) with diagonal = inf.
            calib_all_matrix (if return_all_scores): (N, N, K_max) with diagonal = inf.
        """
        cosines_list = calib_cosines["cosines"]
        gt_indices = calib_cosines["gt_indices"]
        N = len(cosines_list)
 
        if not apply_softmax:
            # Without softmax: extract only landmark-to-landmark scores
            # cosines_list[j][i, gt_indices[j]] = cosine(source_i, true_label_j)
            calib_true = np.empty((N, N), dtype=np.float64)
            for j in range(N):
                calib_true[:, j] = 1.0 - cosines_list[j][:, gt_indices[j]]
            calib_true[np.arange(N), np.arange(N)] = np.inf
            if return_all_scores:
                raise NotImplementedError(
                    "return_all_scores requires apply_softmax=True."
                )
            return calib_true
 
        # With softmax
        calib_true = np.empty((N, N), dtype=np.float64)
        calib_all_list = [] if return_all_scores else None
 
        for j in range(N):
            cos = torch.from_numpy(cosines_list[j])  # (N, K_j)
            cos[j, :] = torch.finfo(cos.dtype).min     # mask self
            probs = torch.softmax(cos / temperature, dim=1)
            scores = 1.0 - probs  # (N, K_j)
 
            calib_true[:, j] = scores[:, gt_indices[j]].numpy()
 
            if return_all_scores:
                calib_all_list.append(scores.numpy())
 
            del cos, probs, scores
 
        np.fill_diagonal(calib_true, np.inf)
 
        if not return_all_scores:
            return calib_true
 
        # Build padded (N, N, K_max) matrix
        K_max = max(arr.shape[1] for arr in calib_all_list)
        mem_gb = estimate_memory_gb((N, N, K_max))
        logger.info(f"Allocating calib_all_matrix: ({N}, {N}, {K_max}) = {mem_gb:.1f} GB")
 
        calib_all_matrix = np.full((N, N, K_max), 1.1)
        for j, scores_j in enumerate(calib_all_list):
            K_j = scores_j.shape[1]
            calib_all_matrix[:, j, :K_j] = scores_j
        for i in range(N):
            calib_all_matrix[i, i, :] = np.inf
 
        del calib_all_list
        gc.collect()
 
        return calib_true, calib_all_matrix
        
    @staticmethod
    def apply_eval_scoring(
        eval_cosines: dict,
        temperature: float = 1.0,
        apply_softmax: bool = True,
    ) -> list[np.ndarray]:
        """
        Convert raw evaluation cosines into nonconformity scores.
 
        Args:
            eval_cosines: Dict from compute_eval_cosines.
            temperature: Softmax temperature.
            apply_softmax: Whether to apply softmax normalization.
 
        Returns:
            List of M arrays, each (K_m, N).
        """
        cosines_list = eval_cosines["cosines"]
        all_scores = []
 
        for cos_np in cosines_list:
            cos = torch.from_numpy(cos_np)  # (N, K_m)
 
            if apply_softmax:
                cos = torch.softmax(cos / temperature, dim=1)
 
            scores = 1.0 - cos
            all_scores.append(scores.T.numpy())  # (K_m, N)
            del cos
 
        return all_scores
    
    # ==================================================================
    # Providing an interface to directly call generate calib scores 
    # and generate eval scores when you don't need to worry about caching
    # ==================================================================
 
    def generate_calib_scores(
        self,
        embeddings: list[torch.Tensor],
        landmarks: list[list[float]],
        image_dims: list[tuple[int, int]],
        temperature: float = 1.0,
        apply_softmax: bool = True,
        return_all_scores: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Compute calibration nonconformity scores in one step.
 
        Combines compute_calib_cosines + apply_calib_scoring.
 
        Args:
            embeddings: List of N tensors, each (K_j, D).
            landmarks: List of N [x, y] coordinates.
            temperature: Softmax temperature.
            apply_softmax: Whether to apply softmax normalization.
            return_all_scores: If True, also return (N, N, K_max) matrix.
 
        Returns:
            calib_true_matrix: (N, N) with diagonal = inf.
            calib_all_matrix (if return_all_scores): (N, N, K_max).
        """
        cosines = self.compute_calib_cosines(embeddings, landmarks, image_dims)
        return self.apply_calib_scoring(
            cosines, temperature=temperature,
            apply_softmax=apply_softmax,
            return_all_scores=return_all_scores,
        )
 
    def generate_eval_scores(
        self,
        test_embeddings: list[torch.Tensor],
        calib_lm_embeddings: list[torch.Tensor],
        temperature: float = 1.0,
        apply_softmax: bool = True,
    ) -> list[np.ndarray]:
        """
        Compute evaluation nonconformity scores in one step.
 
        Combines compute_eval_cosines + apply_eval_scoring.
 
        Args:
            test_embeddings: List of M tensors, each (K_m, D).
            calib_lm_embeddings: List of N tensors, each (D,).
            temperature: Softmax temperature.
            apply_softmax: Whether to apply softmax normalization.
 
        Returns:
            List of M arrays, each (K_m, N).
        """
        cosines = self.compute_eval_cosines(test_embeddings, calib_lm_embeddings)
        return self.apply_eval_scoring(
            cosines, temperature=temperature,
            apply_softmax=apply_softmax,
        )