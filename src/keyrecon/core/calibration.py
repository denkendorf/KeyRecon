from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from ..config import AdaptiveCKSWeights, DEFAULT_THRESHOLD_GRID, EN_REFERENCE_WEIGHTS
from .metrics import evaluate_predictions
from .scoring import score_features, rank_predictions
from .weights import generate_weight_candidates, distance_from_baseline

@dataclass
class SearchResult:
    weights: AdaptiveCKSWeights
    threshold: float
    metrics: dict[str, float]
    search_table: pd.DataFrame

def _evaluate_config(
    oof_features: pd.DataFrame,
    gold: pd.DataFrame,
    weights: AdaptiveCKSWeights,
    threshold: float,
    fold_lookup: dict[str, int] | None,
) -> tuple[dict[str, float], float]:
    scored = score_features(oof_features, weights)
    preds = rank_predictions(scored, threshold, top_k=10)
    metrics, doc = evaluate_predictions(preds, gold, fold_lookup=fold_lookup)
    if fold_lookup and not doc.empty:
        fold_means = doc.groupby("fold_id")["f1_top10"].mean()
        fold_sd = float(fold_means.std(ddof=1)) if len(fold_means) > 1 else 0.0
    else:
        fold_sd = 0.0
    return metrics, fold_sd

def calibrate_threshold(
    oof_features: pd.DataFrame,
    gold: pd.DataFrame,
    *,
    weights: AdaptiveCKSWeights = EN_REFERENCE_WEIGHTS,
    thresholds=DEFAULT_THRESHOLD_GRID,
    fold_lookup: dict[str, int] | None = None,
) -> SearchResult:
    rows = []
    for threshold in thresholds:
        metrics, fold_sd = _evaluate_config(
            oof_features, gold, weights, float(threshold), fold_lookup
        )
        rows.append({
            "threshold": float(threshold),
            **metrics,
            "fold_sd_f1_top10": fold_sd,
        })
    table = pd.DataFrame(rows).sort_values(
        ["f1_top10", "f1_top5", "any_match_top10", "fold_sd_f1_top10", "threshold"],
        ascending=[False, False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    best = table.iloc[0]
    return SearchResult(
        weights=weights,
        threshold=float(best["threshold"]),
        metrics={k: float(best[k]) for k in [
            "precision_top10", "recall_top10", "f1_top10", "f1_top5",
            "any_match_top10", "mean_predictions_top10"
        ]},
        search_table=table,
    )

def adaptive_search(
    oof_features: pd.DataFrame,
    gold: pd.DataFrame,
    *,
    thresholds=DEFAULT_THRESHOLD_GRID,
    fold_lookup: dict[str, int] | None = None,
) -> SearchResult:
    rows = []
    candidates = generate_weight_candidates()
    for idx, weights in enumerate(candidates, start=1):
        scored = score_features(oof_features, weights)
        for threshold in thresholds:
            preds = rank_predictions(scored, float(threshold), top_k=10)
            metrics, doc = evaluate_predictions(preds, gold, fold_lookup=fold_lookup)
            if fold_lookup and not doc.empty:
                fold_means = doc.groupby("fold_id")["f1_top10"].mean()
                fold_sd = float(fold_means.std(ddof=1)) if len(fold_means) > 1 else 0.0
            else:
                fold_sd = 0.0
            rows.append({
                "weight_index": idx,
                "threshold": float(threshold),
                **weights.as_dict(),
                **metrics,
                "fold_sd_f1_top10": fold_sd,
                "distance_from_baseline": distance_from_baseline(weights),
            })
    table = pd.DataFrame(rows).sort_values(
        [
            "f1_top10", "f1_top5", "any_match_top10",
            "fold_sd_f1_top10", "distance_from_baseline", "threshold", "weight_index"
        ],
        ascending=[False, False, False, True, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    best = table.iloc[0]
    weights = AdaptiveCKSWeights(
        tfidf=float(best["tfidf"]),
        df=float(best["df"]),
        dispersion=float(best["dispersion"]),
        domain_focus=float(best["domain_focus"]),
        phrase_quality=float(best["phrase_quality"]),
    )
    return SearchResult(
        weights=weights,
        threshold=float(best["threshold"]),
        metrics={k: float(best[k]) for k in [
            "precision_top10", "recall_top10", "f1_top10", "f1_top5",
            "any_match_top10", "mean_predictions_top10"
        ]},
        search_table=table,
    )
