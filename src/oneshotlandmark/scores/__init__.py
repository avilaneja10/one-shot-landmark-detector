"""
oneshotlandmark.scores — nonconformity score generators and shared score rules.

The generators here turn embeddings into the calibration/evaluation scores the
conformal methods consume:

  - CosineSoftmaxScoreGenerator : softmax "classification" scores
                              (1 - softmax(cos)), in ``cosine_softmax.py``
                              (``--score cosine``).
  - PerSourceDistanceScoreGenerator : "Scheme A" distance-to-peak, one peak per
                              source, in ``per_source_distance.py``
                              (``--score distance``).
  - FusedDistanceScoreGenerator : "Scheme B" distance-to-peak, fuse sources into
                              ONE peak first, in ``fused_distance.py``
                              (``--score distance_fused``). See ``cp_k_for_score``
                              for the k=1 rule it requires.
"""
import logging

logger = logging.getLogger(__name__)

# Scoring families that fuse the sources INSIDE the generator (Scheme B). For
# these the CP stage must NOT aggregate again, so its min-k-mean k is pinned to 1.
_FUSED_SCORES = {"distance_fused"}


def cp_k_for_score(score: str, k: int) -> int:
    """Return the k the CP stage (caos_ragged) must use for a given scoring rule.

    Scheme B ("distance_fused") already does the min-k-mean over sources inside the
    score generator, collapsing the source axis to size 1. Running caos_ragged
    with any k > 1 on that single-source axis would be wrong (double aggregation),
    so we force k=1. Every other scoring rule keeps the k it was given.

    This is the ONE place the "Scheme B => CP k=1" mandate lives; both run.py and
    the viz producer route their CP-stage k through here so neither the CLI --k
    nor a YAML k can break the invariant. The user's k still reaches the Scheme B
    generator as its FUSION k (how many sources to fuse) — only the CP-stage k is
    pinned.
    """
    if score in _FUSED_SCORES:
        if k != 1:
            logger.warning(
                "score=%s fuses sources in the generator; forcing CP k=1 "
                "(ignoring k=%s for the conformal stage).", score, k
            )
        return 1
    return k
