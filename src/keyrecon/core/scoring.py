from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import AdaptiveCKSWeights, DECISION_SCORE_DECIMALS, DEFAULT_TOP_K

FEATURE_MAP = {
    "tfidf": "tfidf_feature",
    "df": "df_feature",
    "dispersion": "dispersion_feature",
    "domain_focus": "domain_focus_feature",
    "phrase_quality": "phrase_quality_feature",
}

def score_features(features: pd.DataFrame, weights: AdaptiveCKSWeights) -> pd.DataFrame:
    work = features.copy()
    raw = np.zeros(len(work), dtype=float)
    for key, weight in weights.as_dict().items():
        col = FEATURE_MAP[key]
        if col not in work.columns:
            raise ValueError(f"Missing feature column: {col}")
        raw += float(weight) * pd.to_numeric(work[col], errors="raise").to_numpy(dtype=float)
    work["cks_score_raw"] = raw
    work["cks_score"] = np.round(raw, DECISION_SCORE_DECIMALS)
    return work

def rank_predictions(
    scored: pd.DataFrame,
    threshold: float,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> pd.DataFrame:
    """Inclusive thresholding after 12-decimal decision-score rounding."""
    if scored.empty:
        return scored.assign(rank=pd.Series(dtype=int))
    required = {"record_id", "canonical_key", "cks_score"}
    missing = required - set(scored.columns)
    if missing:
        raise ValueError(f"Scored candidate table lacks {sorted(missing)}")

    work = scored.loc[pd.to_numeric(scored["cks_score"]).ge(float(threshold))].copy()
    work = work.sort_values(
        ["record_id", "cks_score", "canonical_key"],
        ascending=[True, False, True],
        kind="stable",
    )
    work["rank"] = work.groupby("record_id").cumcount() + 1
    return work.loc[work["rank"] <= int(top_k)].reset_index(drop=True)
