from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

DECISION_SCORE_DECIMALS = 12
DEFAULT_TOP_K = 10
DEFAULT_THRESHOLD_GRID = tuple(round(i * 0.05, 2) for i in range(13))
WEIGHT_SEARCH_SEED = 20260712

BASELINE_WEIGHTS = {
    "tfidf": 0.35,
    "df": 0.20,
    "dispersion": 0.20,
    "domain_focus": 0.15,
    "phrase_quality": 0.10,
}

@dataclass(frozen=True)
class AdaptiveCKSWeights:
    tfidf: float
    df: float
    dispersion: float
    domain_focus: float
    phrase_quality: float

    def __post_init__(self) -> None:
        values = self.as_tuple()
        if any(v < 0 for v in values):
            raise ValueError("Adaptive CKS weights must be non-negative.")
        if abs(sum(values) - 1.0) > 1e-9:
            raise ValueError(f"Adaptive CKS weights must sum to one; got {sum(values):.15f}.")

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return (
            float(self.tfidf),
            float(self.df),
            float(self.dispersion),
            float(self.domain_focus),
            float(self.phrase_quality),
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "tfidf": float(self.tfidf),
            "df": float(self.df),
            "dispersion": float(self.dispersion),
            "domain_focus": float(self.domain_focus),
            "phrase_quality": float(self.phrase_quality),
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> "AdaptiveCKSWeights":
        return cls(
            tfidf=float(values["tfidf"]),
            df=float(values["df"]),
            dispersion=float(values["dispersion"]),
            domain_focus=float(values["domain_focus"]),
            phrase_quality=float(values["phrase_quality"]),
        )

EN_REFERENCE_WEIGHTS = AdaptiveCKSWeights(
    tfidf=0.3588301778033513,
    df=0.1395506716416042,
    dispersion=0.1232507796095620,
    domain_focus=0.2152185351946644,
    phrase_quality=0.1631498357508180,
)
