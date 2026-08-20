from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable
import numpy as np
import pandas as pd

BOUNDARY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "for", "from", "in", "into", "is", "of", "on", "or", "that", "the",
    "their", "this", "to", "with", "without",
}
GENERIC_ACADEMIC_WORDS = {
    "analysis", "approach", "article", "data", "effect", "evidence",
    "finding", "method", "paper", "result", "study",
}

def english_reference_structural_quality(phrase: object) -> float:
    """Frozen English structural H(c) used by the reference profile."""
    import re
    tokens = str(phrase).casefold().split()
    if not tokens:
        return 0.0
    score = 1.0
    alphabetic_like = [bool(re.fullmatch(r"[a-z][a-z0-9'-]*", t)) for t in tokens]
    score *= 0.50 + 0.50 * float(np.mean(alphabetic_like))
    if tokens[0] in BOUNDARY_STOPWORDS:
        score *= 0.35
    if tokens[-1] in BOUNDARY_STOPWORDS:
        score *= 0.35
    if any(a == b for a, b in zip(tokens, tokens[1:])):
        score *= 0.60
    if len(tokens) == 1 and tokens[0] in GENERIC_ACADEMIC_WORDS:
        score *= 0.45
    if any(len(t) == 1 and t not in {"x", "y"} for t in tokens):
        score *= 0.75
    return float(np.clip(score, 0.0, 1.0))

def generic_unicode_structural_quality(phrase: object) -> float:
    """Language-neutral experimental structural heuristic.

    This is intentionally not described as a validated replacement for the
    English reference PhraseQuality heuristic.
    """
    tokens = str(phrase).split()
    if not tokens:
        # CJK canonical keys may contain no spaces; non-empty text is still one expression.
        tokens = [str(phrase)] if str(phrase).strip() else []
    if not tokens:
        return 0.0
    score = 1.0
    content_ratio = np.mean([
        bool(t.strip()) and any(ch.isalnum() for ch in t)
        for t in tokens
    ])
    score *= 0.50 + 0.50 * float(content_ratio)
    if any(a == b for a, b in zip(tokens, tokens[1:])):
        score *= 0.60
    return float(np.clip(score, 0.0, 1.0))

def minmax_within_group(values: pd.Series, groups: pd.Series) -> pd.Series:
    frame = pd.DataFrame({
        "value": pd.to_numeric(values, errors="coerce").fillna(0.0),
        "group": groups.astype(str),
    })
    low = frame.groupby("group")["value"].transform("min")
    high = frame.groupby("group")["value"].transform("max")
    denom = high - low
    scaled = np.where(
        denom.gt(0),
        (frame["value"] - low) / denom,
        np.where(high.gt(0), 1.0, 0.0),
    )
    return pd.Series(scaled, index=values.index, dtype=float)

def prepare_candidate_frame(candidate_table: pd.DataFrame) -> pd.DataFrame:
    work = candidate_table.copy()
    required = {"record_id", "canonical_key", "candidate_length", "total_occurrence_count"}
    missing = required - set(work.columns)
    if missing:
        raise ValueError(f"Candidate table lacks {sorted(missing)}")
    work["record_id"] = work["record_id"].astype(str)
    work["canonical_key"] = work["canonical_key"].astype(str)
    work["canonical_length"] = pd.to_numeric(work["candidate_length"], errors="raise").astype(int)
    work["count_in_doc"] = pd.to_numeric(work["total_occurrence_count"], errors="raise").astype(float)
    work["count_sum_doc_length"] = (
        work.groupby(["record_id", "canonical_length"])["count_in_doc"].transform("sum")
    )
    work["tf"] = (
        work["count_in_doc"] / work["count_sum_doc_length"].replace(0, np.nan)
    ).fillna(0.0)
    return work

@dataclass
class FeatureResources:
    fit_record_count: int
    df: dict[str, int]
    dispersion: dict[str, float]
    domain_df: dict[str, int]
    max_domain_df: int
    length_prior: dict[int, float]

def fit_feature_resources(
    candidate_table: pd.DataFrame,
    fit_ids: set[str],
    gold_table: pd.DataFrame,
) -> FeatureResources:
    work = prepare_candidate_frame(candidate_table)
    fit_ids = {str(x) for x in fit_ids}
    n = len(fit_ids)
    if n < 2:
        raise ValueError("At least two fit records are required.")
    fit_candidates = work.loc[work["record_id"].isin(fit_ids)].copy()

    df = fit_candidates.groupby("canonical_key")["record_id"].nunique().astype(int).to_dict()

    disp_doc = fit_candidates[["record_id", "canonical_key", "count_in_doc"]].copy()
    totals = disp_doc.groupby("canonical_key")["count_in_doc"].transform("sum")
    disp_doc["share"] = (disp_doc["count_in_doc"] / totals.replace(0, np.nan)).fillna(0.0)
    disp_doc["entropy_part"] = np.where(
        disp_doc["share"].gt(0),
        -disp_doc["share"] * np.log(disp_doc["share"]),
        0.0,
    )
    disp_agg = disp_doc.groupby("canonical_key").agg(
        entropy=("entropy_part", "sum"),
        observed_documents=("record_id", "nunique"),
    )
    disp_agg["value"] = np.where(
        disp_agg["observed_documents"].gt(1),
        disp_agg["entropy"] / math.log(n),
        0.0,
    )
    dispersion = disp_agg["value"].clip(0.0, 1.0).to_dict()

    gold = gold_table.copy()
    gold["record_id"] = gold["record_id"].astype(str)
    fit_gold = gold.loc[gold["record_id"].isin(fit_ids)].copy()
    fit_gold = fit_gold.drop_duplicates(["record_id", "gold_canonical_key"])
    domain_df = fit_gold.groupby("gold_canonical_key")["record_id"].nunique().astype(int).to_dict()
    max_domain_df = max(domain_df.values(), default=1)

    if "gold_length_bin" not in fit_gold.columns:
        fit_gold["gold_length_bin"] = (
            fit_gold["gold_canonical_key"].astype(str).str.split().str.len().clip(upper=5).astype(int)
        )
    counts = fit_gold["gold_length_bin"].astype(int).value_counts().to_dict()
    max_count = max(counts.values(), default=1)
    length_prior = {int(k): float(v / max_count) for k, v in counts.items()}

    return FeatureResources(
        fit_record_count=n,
        df={str(k): int(v) for k, v in df.items()},
        dispersion={str(k): float(v) for k, v in dispersion.items()},
        domain_df={str(k): int(v) for k, v in domain_df.items()},
        max_domain_df=int(max_domain_df),
        length_prior=length_prior,
    )

def transform_features(
    candidate_table: pd.DataFrame,
    resources: FeatureResources,
    *,
    structural_quality: Callable[[object], float] = english_reference_structural_quality,
) -> pd.DataFrame:
    work = prepare_candidate_frame(candidate_table)
    n = int(resources.fit_record_count)
    work["df_fit"] = work["canonical_key"].map(resources.df).fillna(0).astype(int)
    work["idf_fit"] = np.log((n + 1) / (work["df_fit"] + 1)) + 1.0
    work["tfidf_raw"] = work["tf"] * work["idf_fit"]
    work["tfidf_feature"] = minmax_within_group(work["tfidf_raw"], work["record_id"])
    work["df_feature"] = (np.log1p(work["df_fit"]) / math.log1p(n)).clip(0.0, 1.0)
    work["dispersion_feature"] = (
        work["canonical_key"].map(resources.dispersion).fillna(0.0).clip(0.0, 1.0)
    )
    work["domain_gold_df_fit"] = work["canonical_key"].map(resources.domain_df).fillna(0).astype(int)
    work["domain_focus_feature"] = (
        np.log1p(work["domain_gold_df_fit"]) / math.log1p(max(resources.max_domain_df, 1))
    ).clip(0.0, 1.0)
    work["phrase_structural_quality"] = work["canonical_key"].map(structural_quality)
    work["phrase_length_prior"] = work["canonical_length"].map(resources.length_prior).fillna(0.0)
    work["phrase_quality_feature"] = (
        0.70 * work["phrase_structural_quality"] + 0.30 * work["phrase_length_prior"]
    ).clip(0.0, 1.0)
    return work
