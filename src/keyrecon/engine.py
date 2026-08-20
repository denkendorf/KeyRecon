from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

from .canonical import Canonicalizer
from .candidate_table import aggregate_candidates
from .config import EN_REFERENCE_WEIGHTS, AdaptiveCKSWeights
from .core.calibration import adaptive_search, calibrate_threshold, SearchResult
from .core.features import fit_feature_resources, transform_features, FeatureResources
from .core.scoring import score_features, rank_predictions
from .languages.registry import make_adapter
from .normalization import normalize_exact_key
from .version import __version__

def _keyword_rows(
    records: pd.DataFrame,
    adapter,
    canonicalizer: Canonicalizer,
    *,
    id_col: str,
    keywords_col: str,
    delimiter: str,
) -> pd.DataFrame:
    rows = []
    for rid, raw in records[[id_col, keywords_col]].itertuples(index=False, name=None):
        if pd.isna(raw) or not str(raw).strip():
            continue
        for piece in str(raw).split(delimiter):
            piece = piece.strip()
            if not piece:
                continue
            canonical = canonicalizer(piece)
            rows.append({
                "record_id": str(rid),
                "gold_surface": piece,
                "gold_canonical_key": canonical,
                "gold_length_bin": min(5, adapter.gold_length(canonical)),
            })
    return pd.DataFrame(rows).drop_duplicates(["record_id", "gold_canonical_key"])

def _deterministic_folds(record_ids: list[str], n_folds: int, seed: int) -> dict[str, int]:
    ids = np.array(sorted(set(map(str, record_ids))), dtype=object)
    if len(ids) < n_folds:
        raise ValueError(f"Need at least {n_folds} observed records; got {len(ids)}.")
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    return {str(rid): int(i % n_folds) for i, rid in enumerate(ids)}

@dataclass
class FitSummary:
    observed_records: int
    candidate_rows: int
    gold_rows: int
    mode: str
    selected_weights: dict[str, float]
    selected_threshold: float
    oof_metrics: dict[str, float]

class KeyRecon:
    """End-to-end author-keyword reconstruction.

    The English reference profile mirrors the published/frozen candidate and
    scoring architecture. Chinese, Japanese, and Korean adapters are experimental
    and should be independently evaluated before substantive use.
    """

    def __init__(
        self,
        *,
        language: str = "en",
        model_name: str | None = None,
        canonicalizer: Canonicalizer | None = None,
        mode: str = "adaptive",
        n_folds: int = 3,
        fold_seed: int = 636,
        strict_reference: bool = False,
    ) -> None:
        if mode not in {"adaptive", "reference"}:
            raise ValueError("mode must be 'adaptive' or 'reference'")
        self.language = language.lower()
        self.adapter = make_adapter(
            self.language, model_name=model_name, strict_reference=strict_reference
        )
        if mode == "reference" and self.language != "en":
            raise ValueError("mode='reference' is available only for the English en_reference profile.")
        self.canonicalizer = canonicalizer or Canonicalizer.identity()
        self.mode = mode
        self.n_folds = int(n_folds)
        self.fold_seed = int(fold_seed)
        self.weights_: AdaptiveCKSWeights | None = None
        self.threshold_: float | None = None
        self.resources_: FeatureResources | None = None
        self.fit_summary_: FitSummary | None = None
        self.search_table_: pd.DataFrame | None = None
        self._fit_ids: set[str] = set()

    def fit(
        self,
        records: pd.DataFrame,
        *,
        id_col: str = "record_id",
        title_col: str = "title",
        abstract_col: str = "abstract",
        keywords_col: str = "author_keywords",
        delimiter: str = ";",
        batch_size: int = 8,
    ) -> "KeyRecon":
        required = {id_col, title_col, abstract_col, keywords_col}
        missing = required - set(records.columns)
        if missing:
            raise ValueError(f"Input records lack {sorted(missing)}")
        work = records.copy()
        work[id_col] = work[id_col].astype(str)
        observed_mask = work[keywords_col].fillna("").astype(str).str.strip().ne("")
        observed = work.loc[observed_mask].copy()
        if len(observed) < max(self.n_folds, 3):
            raise ValueError("Too few records with observed author keywords to fit KeyRecon.")

        gold = _keyword_rows(
            observed, self.adapter, self.canonicalizer,
            id_col=id_col, keywords_col=keywords_col, delimiter=delimiter,
        )
        if gold.empty:
            raise ValueError("No usable observed author keywords were parsed.")

        occ = self.adapter.generate_candidates(
            observed, id_col=id_col, title_col=title_col,
            abstract_col=abstract_col, batch_size=batch_size,
        )
        candidates = aggregate_candidates(occ, self.canonicalizer, self.adapter)
        if candidates.empty:
            raise ValueError("Candidate generation returned no candidates.")

        fit_ids = set(observed[id_col].astype(str))
        fold_lookup = _deterministic_folds(sorted(fit_ids), self.n_folds, self.fold_seed)

        oof_parts = []
        for heldout_fold in range(self.n_folds):
            heldout_ids = {rid for rid, fold in fold_lookup.items() if fold == heldout_fold}
            train_ids = fit_ids - heldout_ids
            resources = fit_feature_resources(candidates, train_ids, gold)
            heldout_cands = candidates.loc[candidates["record_id"].isin(heldout_ids)].copy()
            transformed = transform_features(
                heldout_cands, resources, structural_quality=self.adapter.structural_quality
            )
            transformed["fold_id"] = heldout_fold
            oof_parts.append(transformed)
        oof_features = pd.concat(oof_parts, ignore_index=True)

        if self.mode == "adaptive":
            result = adaptive_search(oof_features, gold, fold_lookup=fold_lookup)
        else:
            result = calibrate_threshold(
                oof_features, gold, weights=EN_REFERENCE_WEIGHTS, fold_lookup=fold_lookup
            )

        self.weights_ = result.weights
        self.threshold_ = float(result.threshold)
        self.search_table_ = result.search_table.copy()
        self.resources_ = fit_feature_resources(candidates, fit_ids, gold)
        self._fit_ids = fit_ids
        self.fit_summary_ = FitSummary(
            observed_records=len(fit_ids),
            candidate_rows=len(candidates),
            gold_rows=len(gold),
            mode=self.mode,
            selected_weights=self.weights_.as_dict(),
            selected_threshold=self.threshold_,
            oof_metrics=result.metrics,
        )
        return self

    def reconstruct(
        self,
        records: pd.DataFrame,
        *,
        id_col: str = "record_id",
        title_col: str = "title",
        abstract_col: str = "abstract",
        batch_size: int = 8,
        top_k: int = 10,
    ) -> pd.DataFrame:
        if self.resources_ is None or self.weights_ is None or self.threshold_ is None:
            raise RuntimeError("Call fit() before reconstruct().")
        work = records.copy()
        work[id_col] = work[id_col].astype(str)
        occ = self.adapter.generate_candidates(
            work, id_col=id_col, title_col=title_col,
            abstract_col=abstract_col, batch_size=batch_size,
        )
        candidates = aggregate_candidates(occ, self.canonicalizer, self.adapter)
        if candidates.empty:
            return pd.DataFrame(columns=["record_id", "canonical_key", "cks_score", "rank"])
        features = transform_features(
            candidates, self.resources_, structural_quality=self.adapter.structural_quality
        )
        scored = score_features(features, self.weights_)
        return rank_predictions(scored, self.threshold_, top_k=top_k)

    def fit_reconstruct_missing(
        self,
        records: pd.DataFrame,
        *,
        id_col: str = "record_id",
        title_col: str = "title",
        abstract_col: str = "abstract",
        keywords_col: str = "author_keywords",
        delimiter: str = ";",
        batch_size: int = 8,
        top_k: int = 10,
    ) -> pd.DataFrame:
        self.fit(
            records,
            id_col=id_col, title_col=title_col, abstract_col=abstract_col,
            keywords_col=keywords_col, delimiter=delimiter, batch_size=batch_size,
        )
        missing_mask = records[keywords_col].fillna("").astype(str).str.strip().eq("")
        targets = records.loc[missing_mask].copy()
        return self.reconstruct(
            targets, id_col=id_col, title_col=title_col,
            abstract_col=abstract_col, batch_size=batch_size, top_k=top_k,
        )

    def manifest(self) -> dict:
        if self.fit_summary_ is None:
            raise RuntimeError("Call fit() first.")
        return {
            "keyrecon_version": __version__,
            "language": self.language,
            "language_adapter": self.adapter.runtime_metadata(),
            "mode": self.mode,
            "selected_weights": self.fit_summary_.selected_weights,
            "selected_threshold": self.fit_summary_.selected_threshold,
            "threshold_source": "development-only out-of-fold calibration",
            "decision_score_round_decimals": 12,
            "threshold_rule": "inclusive: rounded score >= threshold",
            "top_k_default": 10,
            "observed_fit_records": self.fit_summary_.observed_records,
            "oof_metrics": self.fit_summary_.oof_metrics,
            "canonicalization": (
                "identity exact normalization" if not self.canonicalizer.mapping
                else "user-supplied performance-blind mapping"
            ),
        }
