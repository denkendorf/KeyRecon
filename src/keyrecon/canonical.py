from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from .normalization import normalize_exact_key

@dataclass
class Canonicalizer:
    """Performance-blind mapping interface.

    By default, canonicalization is identity after exact normalization.
    A user-supplied mapping may be loaded for a domain-specific study.
    """

    mapping: dict[str, str]

    @classmethod
    def identity(cls) -> "Canonicalizer":
        return cls(mapping={})

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        source_col: str = "candidate_exact_key",
        target_col: str = "canonical_key",
    ) -> "Canonicalizer":
        df = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
        if source_col not in df.columns or target_col not in df.columns:
            raise ValueError(f"Mapping CSV must contain {source_col!r} and {target_col!r}.")
        mapping = {
            normalize_exact_key(s): normalize_exact_key(t)
            for s, t in zip(df[source_col], df[target_col])
            if normalize_exact_key(s)
        }
        return cls(mapping=mapping)

    def __call__(self, value: object) -> str:
        key = normalize_exact_key(value)
        return self.mapping.get(key, key)
