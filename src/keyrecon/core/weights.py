from __future__ import annotations

import numpy as np
from ..config import (
    AdaptiveCKSWeights,
    BASELINE_WEIGHTS,
    WEIGHT_SEARCH_SEED,
)

WEIGHT_KEYS = ("tfidf", "df", "dispersion", "domain_focus", "phrase_quality")

def generate_weight_candidates(seed: int = WEIGHT_SEARCH_SEED) -> list[AdaptiveCKSWeights]:
    """Reproduce the prespecified 61-vector Adaptive CKS search space."""
    baseline = np.array([BASELINE_WEIGHTS[k] for k in WEIGHT_KEYS], dtype=float)
    candidates = [baseline.copy()]
    step = 0.05
    for source in range(len(WEIGHT_KEYS)):
        for target in range(len(WEIGHT_KEYS)):
            if source == target:
                continue
            v = baseline.copy()
            if v[source] >= step:
                v[source] -= step
                v[target] += step
                candidates.append(v)

    rng = np.random.default_rng(seed)
    alpha = baseline * 50.0 + 1.0
    for _ in range(40):
        candidates.append(rng.dirichlet(alpha))

    unique: dict[tuple[float, ...], np.ndarray] = {}
    for v in candidates:
        v = np.clip(v, 0, None)
        v = v / v.sum()
        unique[tuple(np.round(v, 6))] = v

    result = [
        AdaptiveCKSWeights(**{k: float(x) for k, x in zip(WEIGHT_KEYS, v)})
        for v in unique.values()
    ]
    if len(result) != 61:
        raise RuntimeError(f"Weight-search design drift: expected 61 vectors, got {len(result)}")
    return result

def distance_from_baseline(weights: AdaptiveCKSWeights) -> float:
    return float(sum(
        abs(weights.as_dict()[k] - BASELINE_WEIGHTS[k])
        for k in WEIGHT_KEYS
    ))
