from __future__ import annotations

from collections import defaultdict
import pandas as pd

def set_metrics(predicted: list[str], gold_values: list[str]) -> dict[str, float]:
    predicted_set = {v for v in predicted if v}
    gold_set = {v for v in gold_values if v}
    matches = len(predicted_set & gold_set)
    precision = matches / len(predicted_set) if predicted_set else 0.0
    recall = matches / len(gold_set) if gold_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = len(predicted_set | gold_set)
    return {
        "matches": float(matches),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "jaccard": float(matches / union if union else 0.0),
        "any_match": float(matches > 0),
        "predicted_count": float(len(predicted_set)),
        "gold_count": float(len(gold_set)),
    }

def evaluate_predictions(
    predictions: pd.DataFrame,
    gold: pd.DataFrame,
    *,
    record_col: str = "record_id",
    pred_col: str = "canonical_key",
    gold_col: str = "gold_canonical_key",
    rank_col: str = "rank",
    top_k: int = 10,
    secondary_k: int = 5,
    fold_lookup: dict[str, int] | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    pred_lookup: dict[str, list[str]] = defaultdict(list)
    if not predictions.empty:
        p = predictions.sort_values([record_col, rank_col], kind="stable")
        for rid, group in p.groupby(record_col, sort=True):
            pred_lookup[str(rid)] = group.loc[group[rank_col] <= top_k, pred_col].astype(str).tolist()

    gold_lookup: dict[str, list[str]] = defaultdict(list)
    for rid, group in gold.groupby(record_col, sort=True):
        gold_lookup[str(rid)] = group[gold_col].astype(str).tolist()

    rows = []
    for rid in sorted(gold_lookup):
        preds = pred_lookup.get(rid, [])
        m10 = set_metrics(preds[:top_k], gold_lookup[rid])
        m5 = set_metrics(preds[:secondary_k], gold_lookup[rid])
        row = {
            "record_id": rid,
            "precision_top10": m10["precision"],
            "recall_top10": m10["recall"],
            "f1_top10": m10["f1"],
            "any_match_top10": m10["any_match"],
            "predicted_count_top10": m10["predicted_count"],
            "f1_top5": m5["f1"],
        }
        if fold_lookup is not None:
            row["fold_id"] = int(fold_lookup[rid])
        rows.append(row)

    doc = pd.DataFrame(rows)
    if doc.empty:
        summary = {
            "precision_top10": 0.0, "recall_top10": 0.0,
            "f1_top10": 0.0, "f1_top5": 0.0,
            "any_match_top10": 0.0, "mean_predictions_top10": 0.0,
        }
    else:
        summary = {
            "precision_top10": float(doc["precision_top10"].mean()),
            "recall_top10": float(doc["recall_top10"].mean()),
            "f1_top10": float(doc["f1_top10"].mean()),
            "f1_top5": float(doc["f1_top5"].mean()),
            "any_match_top10": float(doc["any_match_top10"].mean()),
            "mean_predictions_top10": float(doc["predicted_count_top10"].mean()),
        }
    return summary, doc
