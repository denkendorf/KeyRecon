from __future__ import annotations

import pandas as pd

from .canonical import Canonicalizer
from .languages.base import LanguageAdapter

def aggregate_candidates(
    occurrences: pd.DataFrame,
    canonicalizer: Canonicalizer,
    adapter: LanguageAdapter,
) -> pd.DataFrame:
    if occurrences.empty:
        return pd.DataFrame(columns=[
            "record_id", "canonical_key", "candidate_length",
            "total_occurrence_count", "title_occurrence_count",
            "abstract_occurrence_count", "source_class", "representative_surface",
        ])
    work = occurrences.copy()
    work["canonical_key"] = work["candidate_exact_key"].map(canonicalizer)
    work = work.loc[work["canonical_key"].astype(str).str.strip().ne("")].copy()

    rows = []
    for (rid, canonical), group in work.groupby(["record_id", "canonical_key"], sort=True):
        title_n = int(group["source_field"].eq("TITLE").sum())
        abstract_n = int(group["source_field"].eq("ABSTRACT").sum())
        if title_n and abstract_n:
            source_class = "TITLE_AND_ABSTRACT"
        elif title_n:
            source_class = "TITLE_ONLY"
        else:
            source_class = "ABSTRACT_ONLY"
        surface_counts = (
            group.groupby("candidate_surface").size()
            .sort_values(ascending=False, kind="stable")
        )
        max_count = int(surface_counts.max())
        representative = sorted(surface_counts.loc[surface_counts.eq(max_count)].index.astype(str))[0]
        fallback_length = int(group.loc[group["candidate_surface"].astype(str).eq(representative),
                                        "lexical_unit_count"].iloc[0])
        rows.append({
            "record_id": str(rid),
            "canonical_key": str(canonical),
            "candidate_length": adapter.canonical_length(str(canonical), fallback=fallback_length),
            "total_occurrence_count": int(len(group)),
            "title_occurrence_count": title_n,
            "abstract_occurrence_count": abstract_n,
            "source_class": source_class,
            "representative_surface": representative,
        })
    return pd.DataFrame(rows)
