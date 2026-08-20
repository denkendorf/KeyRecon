from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import pandas as pd

from ..candidates.english_reference import EnglishReferenceCandidateGenerator
from ..candidates.generic import GenericSpacyCandidateGenerator
from ..core.features import (
    english_reference_structural_quality,
    generic_unicode_structural_quality,
)

@dataclass(frozen=True)
class LanguageSpec:
    code: str
    label: str
    model: str
    status: str
    candidate_profile: str
    expected_model_version: str | None = None
    reference_spacy_version: str | None = None

class LanguageAdapter:
    def __init__(
        self,
        spec: LanguageSpec,
        *,
        model_name: str | None = None,
        strict_reference: bool = False,
    ) -> None:
        self.spec = spec
        self.model_name = model_name or spec.model
        self.strict_reference = strict_reference
        self._nlp = None
        if spec.candidate_profile == "en_reference":
            self.generator = EnglishReferenceCandidateGenerator()
            self.structural_quality = english_reference_structural_quality
        else:
            self.generator = GenericSpacyCandidateGenerator()
            self.structural_quality = generic_unicode_structural_quality

    def load(self):
        if self._nlp is not None:
            return self._nlp
        try:
            import spacy
        except ImportError as exc:
            raise RuntimeError(
                "spaCy is required for raw-text candidate generation. "
                "Install KeyRecon with `pip install -e '.[nlp]'` or install spaCy separately."
            ) from exc
        try:
            nlp = spacy.load(self.model_name, disable=["ner"])
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model {self.model_name!r} is not installed. "
                f"Run: python -m spacy download {self.model_name}"
            ) from exc
        if "parser" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
            raise RuntimeError(f"{self.model_name!r} must provide sentence boundaries.")
        actual = str(nlp.meta.get("version", ""))
        expected = self.spec.expected_model_version
        if expected and actual != expected:
            message = f"Model version drift for {self.model_name}: {actual} != reference {expected}"
            if self.strict_reference:
                raise RuntimeError(message)
            warnings.warn(message, RuntimeWarning)
        self._nlp = nlp
        return nlp

    def generate_candidates(
        self,
        records: pd.DataFrame,
        *,
        id_col: str = "record_id",
        title_col: str = "title",
        abstract_col: str = "abstract",
        batch_size: int = 8,
    ) -> pd.DataFrame:
        nlp = self.load()
        required = {id_col, title_col, abstract_col}
        missing = required - set(records.columns)
        if missing:
            raise ValueError(f"Input records lack {sorted(missing)}")
        source_records = []
        for row in records[[id_col, title_col, abstract_col]].itertuples(index=False, name=None):
            rid, title, abstract = row
            source_records.append((str(rid), "TITLE", "" if pd.isna(title) else str(title)))
            source_records.append((str(rid), "ABSTRACT", "" if pd.isna(abstract) else str(abstract)))
        rows = []
        texts = [x[2] for x in source_records]
        for meta, doc in zip(source_records, nlp.pipe(texts, batch_size=batch_size, n_process=1)):
            rid, source_field, _ = meta
            if not doc.text.strip():
                continue
            rows.extend(self.generator.generate_doc(doc, source_field, rid))
        return pd.DataFrame(rows)

    def gold_length(self, text: str) -> int:
        # Frozen English PhraseQuality length prior counted whitespace-delimited canonical units.
        if self.spec.candidate_profile == "en_reference":
            return max(1, len(str(text).strip().split()))
        nlp = self.load()
        doc = nlp(str(text))
        return max(1, sum(not t.is_punct and not t.is_space for t in doc))

    def canonical_length(self, canonical_key: str, fallback: int = 1) -> int:
        if self.spec.candidate_profile == "en_reference":
            return max(1, len(str(canonical_key).strip().split()))
        # For experimental CJK profiles, preserve the source span length where possible.
        return max(1, int(fallback))

    def runtime_metadata(self) -> dict[str, Any]:
        nlp = self.load()
        try:
            import spacy
            spacy_version = spacy.__version__
        except Exception:
            spacy_version = None
        return {
            "language": self.spec.code,
            "language_label": self.spec.label,
            "profile_status": self.spec.status,
            "candidate_profile": self.spec.candidate_profile,
            "spacy_model": self.model_name,
            "spacy_model_version": nlp.meta.get("version"),
            "spacy_version": spacy_version,
        }
