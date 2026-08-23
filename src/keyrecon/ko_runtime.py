from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import html
import importlib.metadata
import json
import math
import re
import unicodedata

import numpy as np
import pandas as pd

from .ko_reference import KoreanReference, load_korean_reference


EXPECTED_KIWI_VERSION = "0.23.2"
EXPECTED_CONFIGURATION_ID = "W034_S45"
EXPECTED_DEVELOPMENT_RECORDS = 748
EXPECTED_FULL_DEV_RESOURCE_KEYS = 642_776
EXPECTED_FULL_EDGE_PAIRS = 64

MAX_INDEX_WINDOW = 12
MAX_CONTAINED_TOKENS = 12
MAX_CHAR_LENGTH = 60
MAX_WHITESPACE_UNITS = 8

GOLD_LENGTH_BIN_CAP = 12
PHRASE_STRUCTURAL_WEIGHT = 0.70
PHRASE_LENGTH_WEIGHT = 0.30

FIVE_COMPONENT_COLUMNS = [
    "tfidf_feature",
    "df_feature",
    "dispersion_feature",
    "domain_focus_feature",
    "phrase_quality_feature",
]

WEIGHT_ORDER = [
    "tfidf",
    "df",
    "dispersion",
    "domain_focus",
    "phrase_quality",
]

NOMINAL_TAGS = {
    "NNG", "NNP", "NNB", "NR", "NP",
    "SL", "SN", "SH", "XR", "XPN", "XSN",
}

RELATIONAL_TAGS = {
    "JKG", "JC",
}

PUNCT_TAGS = {
    "SSO", "SSC", "SO", "SW", "SP",
}

LEXICAL_ANCHOR_TAGS = {
    "NNG", "NNP", "NNB", "NR", "NP",
    "SL", "SN", "SH", "XR",
}

ZERO_WIDTH_RE = re.compile(
    r"[\u200B\u200C\u200D\uFEFF]"
)


@dataclass
class KoreanRunResult:
    source: pd.DataFrame
    token_audit: pd.DataFrame
    field_audit: pd.DataFrame
    candidate_universe: pd.DataFrame
    candidate_record_summary: pd.DataFrame
    features: pd.DataFrame
    predictions_long: pd.DataFrame
    reconstruction_by_record: pd.DataFrame


def _parse_bool(value) -> bool:
    return str(value).strip().lower() in {
        "true", "1", "yes", "y",
    }


def _stable_json(obj) -> str:
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _normalize_ws(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        unicodedata.normalize(
            "NFC",
            str(text or ""),
        ),
    ).strip()


def _canonical_key(text: str) -> str:
    return _normalize_ws(text).casefold()


def _compact_mapping_key(text: str) -> str:
    return re.sub(
        r"\s+",
        "",
        _canonical_key(text),
    )


def _repeated_html_unescape(
    text: str,
    max_rounds: int = 3,
) -> str:
    s = str(text)
    for _ in range(max_rounds):
        new = html.unescape(s)
        if new == s:
            break
        s = new
    return s


def normalize_source_field(text: str) -> str:
    raw = str(text or "")
    s = unicodedata.normalize("NFC", raw)
    s = _repeated_html_unescape(s)
    s = s.replace("\xa0", " ")
    s = ZERO_WIDTH_RE.sub("", s)
    s = (
        s.replace("\r\n", " ")
        .replace("\r", " ")
        .replace("\n", " ")
    )
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _source_pair_sha256(
    title: str,
    abstract: str,
) -> str:
    import hashlib

    payload = (
        str(title).encode("utf-8")
        + b"\x1e"
        + str(abstract).encode("utf-8")
    )
    return hashlib.sha256(payload).hexdigest()


def _minmax_within_group(
    values: pd.Series,
    groups: pd.Series,
) -> pd.Series:
    frame = pd.DataFrame(
        {
            "value": pd.to_numeric(
                values,
                errors="coerce",
            ).fillna(0.0),
            "group": groups.astype(str),
        },
        index=values.index,
    )

    low = frame.groupby("group")["value"].transform("min")
    high = frame.groupby("group")["value"].transform("max")
    denom = high - low

    scaled = np.where(
        denom.gt(0),
        (frame["value"] - low) / denom,
        np.where(
            high.gt(0),
            1.0,
            0.0,
        ),
    )

    return pd.Series(
        scaled,
        index=values.index,
        dtype=float,
    )


def _pos_category(
    tags: tuple[str, ...],
) -> str:
    tag_set = set(tags)

    if not (tag_set & LEXICAL_ANCHOR_TAGS):
        return "extended_grammar"

    if tag_set <= NOMINAL_TAGS:
        return "nominal_only"

    if tag_set <= (NOMINAL_TAGS | RELATIONAL_TAGS):
        return "nominal_relational"

    if tag_set <= (
        NOMINAL_TAGS
        | RELATIONAL_TAGS
        | PUNCT_TAGS
    ):
        return "nominal_relational_punct"

    return "extended_grammar"


def _sentence_boundary_safe(
    tags: tuple[str, ...],
) -> bool:
    for i, tag in enumerate(tags):
        if tag not in {"SF", "SE"}:
            continue

        has_open_before = "SSO" in tags[:i]
        has_close_after = "SSC" in tags[i + 1:]

        if not (
            has_open_before
            and has_close_after
        ):
            return False

    return True


def _final_refit_accept(
    tags: tuple[str, ...],
    full_edge_pairs: set[tuple[str, str]],
) -> bool:
    category = _pos_category(tags)

    nominal_punct = (
        len(tags) <= 8
        and category in {
            "nominal_only",
            "nominal_relational",
            "nominal_relational_punct",
        }
    )

    edge_allowed = (
        (tags[0], tags[-1])
        in full_edge_pairs
    )

    return (
        (nominal_punct or edge_allowed)
        and _sentence_boundary_safe(tags)
    )


def _iter_character_spans(
    text: str,
    token_df: pd.DataFrame,
):
    g = token_df.sort_values(
        "token_seq",
        kind="stable",
    )

    starts = g["start"].to_numpy(dtype=int)
    ends = g["end"].to_numpy(dtype=int)
    tags = g["tag"].to_numpy(dtype=str)

    n_tokens = len(g)
    seen_char_spans = set()

    for i in range(n_tokens):
        span_start = int(starts[i])
        running_end = span_start

        for j in range(
            i,
            min(
                n_tokens,
                i + MAX_INDEX_WINDOW,
            ),
        ):
            running_end = max(
                running_end,
                int(ends[j]),
            )
            span_end = running_end

            if span_end <= span_start:
                continue

            if span_end - span_start > MAX_CHAR_LENGTH:
                break

            span_key = (
                span_start,
                span_end,
            )

            if span_key in seen_char_spans:
                continue

            seen_char_spans.add(span_key)

            surface = " ".join(
                text[span_start:span_end].split()
            )

            if not surface:
                continue

            if len(surface) > MAX_CHAR_LENGTH:
                continue

            if surface.count(" ") + 1 > MAX_WHITESPACE_UNITS:
                continue

            if not re.search(
                r"[가-힣A-Za-z0-9\u3400-\u9FFF]",
                surface,
            ):
                continue

            local_left = int(
                np.searchsorted(
                    starts,
                    span_start,
                    side="left",
                )
            )

            local_right = int(
                np.searchsorted(
                    starts,
                    span_end,
                    side="left",
                )
            )

            local_starts = starts[
                local_left:local_right
            ]
            local_ends = ends[
                local_left:local_right
            ]
            local_tags = tags[
                local_left:local_right
            ]

            keep = (
                (local_ends <= span_end)
                & (local_ends > local_starts)
            )

            contained_tags = local_tags[keep]

            if (
                len(contained_tags) == 0
                or len(contained_tags)
                > MAX_CONTAINED_TOKENS
            ):
                continue

            yield {
                "surface": surface,
                "canonical_key": _canonical_key(
                    surface
                ),
                "compact_mapping_key": (
                    _compact_mapping_key(
                        surface
                    )
                ),
                "tags": tuple(
                    contained_tags.tolist()
                ),
                "pos_sequence": "+".join(
                    contained_tags.tolist()
                ),
                "token_count": len(
                    contained_tags
                ),
            }


def _new_agg_item(
    compact_key: str,
):
    return {
        "surface_counter": Counter(),
        "pos_counter": Counter(),
        "title_count": 0,
        "abstract_count": 0,
        "token_counts": [],
        "compact_key": compact_key,
    }


def _add_occurrence(
    agg: dict,
    occ: dict,
    component: str,
) -> None:
    key = occ["canonical_key"]

    item = agg.setdefault(
        key,
        _new_agg_item(
            occ["compact_mapping_key"]
        ),
    )

    item["surface_counter"][
        occ["surface"]
    ] += 1

    item["pos_counter"][
        occ["pos_sequence"]
    ] += 1

    item[
        f"{component}_count"
    ] += 1

    item["token_counts"].append(
        int(occ["token_count"])
    )


def _aggregated_candidate_rows(
    rid: str,
    agg: dict,
) -> list[dict]:
    rows = []

    for key in sorted(agg.keys()):
        item = agg[key]

        surfaces_ranked = sorted(
            item["surface_counter"].items(),
            key=lambda kv: (
                -kv[1],
                len(kv[0]),
                kv[0],
            ),
        )

        representative_surface = (
            surfaces_ranked[0][0]
        )

        surface_variants = sorted(
            item["surface_counter"].keys()
        )

        pos_ranked = sorted(
            item["pos_counter"].items(),
            key=lambda kv: (
                -kv[1],
                kv[0],
            ),
        )

        primary_pos = pos_ranked[0][0]

        pos_sequences = sorted(
            item["pos_counter"].keys()
        )

        title_count = int(
            item["title_count"]
        )
        abstract_count = int(
            item["abstract_count"]
        )

        rows.append(
            {
                "record_id": rid,
                "candidate_surface": (
                    representative_surface
                ),
                "candidate_canonical_key": key,
                "compact_mapping_key": (
                    item["compact_key"]
                ),
                "title_occurrence_count": (
                    title_count
                ),
                "abstract_occurrence_count": (
                    abstract_count
                ),
                "total_occurrence_count": (
                    title_count
                    + abstract_count
                ),
                "title_present": (
                    title_count > 0
                ),
                "abstract_present": (
                    abstract_count > 0
                ),
                "source_components": (
                    "title|abstract"
                    if (
                        title_count > 0
                        and abstract_count > 0
                    )
                    else (
                        "title"
                        if title_count > 0
                        else "abstract"
                    )
                ),
                "surface_variant_count": len(
                    surface_variants
                ),
                "surface_variants_json": (
                    ""
                    if len(surface_variants) == 1
                    else _stable_json(
                        surface_variants
                    )
                ),
                "primary_pos_sequence": (
                    primary_pos
                ),
                "pos_sequence_count": len(
                    pos_sequences
                ),
                "pos_sequences_json": (
                    _stable_json(
                        pos_sequences
                    )
                ),
                "min_token_count": min(
                    item["token_counts"]
                ),
                "max_token_count": max(
                    item["token_counts"]
                ),
            }
        )

    return rows


def _resolve_input_columns(
    df: pd.DataFrame,
) -> tuple[str, str, str]:
    record_candidates = [
        "record_id",
        "UT",
        "id",
    ]
    title_candidates = [
        "title_text",
        "title",
        "TI",
    ]
    abstract_candidates = [
        "abstract_text",
        "abstract",
        "AB",
    ]

    def resolve(candidates):
        for col in candidates:
            if col in df.columns:
                return col
        raise KeyError(
            "Missing required column; expected one of "
            + repr(candidates)
        )

    return (
        resolve(record_candidates),
        resolve(title_candidates),
        resolve(abstract_candidates),
    )


class KoreanAdaptiveCKS:
    def __init__(
        self,
        reference_root: str | Path | None = None,
    ):
        self.reference: KoreanReference = (
            load_korean_reference(
                reference_root
            )
        )

        kiwi_version = importlib.metadata.version(
            "kiwipiepy"
        )

        required_version = str(
            self.reference.payload[
                "analyzer"
            ][
                "required_version"
            ]
        )

        if kiwi_version != required_version:
            raise RuntimeError(
                f"Kiwi version mismatch: "
                f"{kiwi_version} != {required_version}"
            )

        if kiwi_version != EXPECTED_KIWI_VERSION:
            raise RuntimeError(
                "Korean runtime was built against "
                "Kiwi 0.23.2 only."
            )

        from kiwipiepy import Kiwi
        self.kiwi = Kiwi()

        self._load_runtime_resources()

    def _load_runtime_resources(
        self,
    ) -> None:
        p = self.reference.payload

        self.configuration_id = str(
            p["profile"]["configuration_id"]
        )
        self.threshold = float(
            p["profile"][
                "minimum_cks_score"
            ]
        )
        self.score_decimals = int(
            p["profile"][
                "decision_score_round_decimals"
            ]
        )
        self.top_k = int(
            p["profile"]["top_k"]
        )

        self.weights = np.array(
            [
                float(
                    p["profile"]["weights"][key]
                )
                for key in WEIGHT_ORDER
            ],
            dtype=float,
        )

        if (
            self.configuration_id
            != EXPECTED_CONFIGURATION_ID
        ):
            raise RuntimeError(
                "Unexpected Korean configuration."
            )

        if abs(
            float(self.weights.sum()) - 1.0
        ) > 1e-12:
            raise RuntimeError(
                "Korean profile weights do not "
                "sum to one."
            )

        root = self.reference.root

        edge_path = root / p[
            "candidate_generation"
        ][
            "edge_inventory_asset"
        ][
            "file"
        ]

        self.edge_inventory = pd.read_csv(
            edge_path,
            encoding="utf-8-sig",
            low_memory=False,
            dtype=str,
        )

        allowed = self.edge_inventory.loc[
            self.edge_inventory[
                "frozen_allowed"
            ].map(
                _parse_bool
            )
        ].copy()

        self.full_edge_pairs = set(
            zip(
                allowed["start_pos"],
                allowed["end_pos"],
            )
        )

        if (
            len(self.full_edge_pairs)
            != EXPECTED_FULL_EDGE_PAIRS
        ):
            raise RuntimeError(
                "Full-development edge inventory "
                "does not contain 64 allowed pairs."
            )

        resource_nodes = p[
            "components"
        ][
            "runtime_resources"
        ]

        self.full_dev_resource = pd.read_csv(
            root / resource_nodes[
                "tfidf_df_dispersion"
            ][
                "file"
            ],
            encoding="utf-8-sig",
            low_memory=False,
        )

        self.domain_resource = pd.read_csv(
            root / resource_nodes[
                "domain_focus"
            ][
                "file"
            ],
            encoding="utf-8-sig",
            low_memory=False,
        )

        self.phrase_resource = pd.read_csv(
            root / resource_nodes[
                "phrase_quality"
            ][
                "file"
            ],
            encoding="utf-8-sig",
            low_memory=False,
        )

        for col in [
            "df_fit",
            "total_occurrences_fit",
            "fit_record_count",
        ]:
            self.full_dev_resource[
                col
            ] = pd.to_numeric(
                self.full_dev_resource[col],
                errors="raise",
            ).astype(int)

        self.full_dev_resource[
            "sum_count_log_count_fit"
        ] = pd.to_numeric(
            self.full_dev_resource[
                "sum_count_log_count_fit"
            ],
            errors="raise",
        ).astype(float)

        if (
            len(self.full_dev_resource)
            != EXPECTED_FULL_DEV_RESOURCE_KEYS
        ):
            raise RuntimeError(
                "K12 full-development resource "
                "key-count drift."
            )

        if set(
            self.full_dev_resource[
                "fit_record_count"
            ].unique()
        ) != {
            EXPECTED_DEVELOPMENT_RECORDS
        }:
            raise RuntimeError(
                "K12 corpus resource fit-record "
                "count drift."
            )

        self.domain_resource[
            "domain_gold_df_fit"
        ] = pd.to_numeric(
            self.domain_resource[
                "domain_gold_df_fit"
            ],
            errors="raise",
        ).astype(int)

        self.domain_resource[
            "max_domain_gold_df_fit"
        ] = pd.to_numeric(
            self.domain_resource[
                "max_domain_gold_df_fit"
            ],
            errors="raise",
        ).astype(int)

        self.domain_resource[
            "fit_record_count"
        ] = pd.to_numeric(
            self.domain_resource[
                "fit_record_count"
            ],
            errors="raise",
        ).astype(int)

        if set(
            self.domain_resource[
                "fit_record_count"
            ].unique()
        ) != {
            EXPECTED_DEVELOPMENT_RECORDS
        }:
            raise RuntimeError(
                "K12 DomainFocus fit-record "
                "count drift."
            )

        self.domain_map = dict(
            zip(
                self.domain_resource[
                    "gold_canonical_key"
                ].astype(str),
                self.domain_resource[
                    "domain_gold_df_fit"
                ].astype(int),
            )
        )

        max_domain_values = self.domain_resource[
            "max_domain_gold_df_fit"
        ].unique()

        if len(max_domain_values) != 1:
            raise RuntimeError(
                "K12 DomainFocus max-DF is "
                "not unique."
            )

        self.max_domain_df = int(
            max_domain_values[0]
        )

        self.phrase_resource[
            "support_fit"
        ] = pd.to_numeric(
            self.phrase_resource[
                "support_fit"
            ],
            errors="raise",
        ).astype(int)

        self.phrase_resource[
            "prior_value"
        ] = pd.to_numeric(
            self.phrase_resource[
                "prior_value"
            ],
            errors="raise",
        ).astype(float)

        self.phrase_resource[
            "fit_record_count"
        ] = pd.to_numeric(
            self.phrase_resource[
                "fit_record_count"
            ],
            errors="raise",
        ).astype(int)

        if set(
            self.phrase_resource[
                "fit_record_count"
            ].unique()
        ) != {
            EXPECTED_DEVELOPMENT_RECORDS
        }:
            raise RuntimeError(
                "K12 PhraseQuality fit-record "
                "count drift."
            )

        pos_resource = (
            self.phrase_resource.loc[
                self.phrase_resource[
                    "resource_type"
                ].eq(
                    "pos_category"
                )
            ]
            .copy()
        )

        length_resource = (
            self.phrase_resource.loc[
                self.phrase_resource[
                    "resource_type"
                ].eq(
                    "length_bin"
                )
            ]
            .copy()
        )

        self.category_support = dict(
            zip(
                pos_resource[
                    "resource_key"
                ].astype(str),
                pos_resource[
                    "support_fit"
                ].astype(int),
            )
        )

        self.category_prior = dict(
            zip(
                pos_resource[
                    "resource_key"
                ].astype(str),
                pos_resource[
                    "prior_value"
                ].astype(float),
            )
        )

        self.length_support = {
            int(key): int(value)
            for key, value in zip(
                length_resource[
                    "resource_key"
                ],
                length_resource[
                    "support_fit"
                ],
            )
        }

        self.length_prior = {
            int(key): float(value)
            for key, value in zip(
                length_resource[
                    "resource_key"
                ],
                length_resource[
                    "prior_value"
                ],
            )
        }

    def _prepare_source(
        self,
        input_data: pd.DataFrame,
    ) -> pd.DataFrame:
        source = input_data.copy()

        record_col, title_col, abstract_col = (
            _resolve_input_columns(source)
        )

        out = pd.DataFrame(
            {
                "record_id": (
                    source[
                        record_col
                    ].astype(str)
                ),
                "title_text": (
                    source[
                        title_col
                    ].fillna(
                        ""
                    ).map(
                        normalize_source_field
                    )
                ),
                "abstract_text": (
                    source[
                        abstract_col
                    ].fillna(
                        ""
                    ).map(
                        normalize_source_field
                    )
                ),
            }
        )

        if "DE_original" in source.columns:
            out["DE_original"] = (
                source[
                    "DE_original"
                ].fillna(
                    ""
                ).astype(str)
            )
        elif "DE" in source.columns:
            out["DE_original"] = (
                source[
                    "DE"
                ].fillna(
                    ""
                ).astype(str)
            )
        else:
            out["DE_original"] = ""

        if out["record_id"].duplicated().any():
            raise RuntimeError(
                "record_id must be unique."
            )

        if out["record_id"].str.len().eq(0).any():
            raise RuntimeError(
                "record_id must be non-empty."
            )

        out["source_pair_sha256"] = [
            _source_pair_sha256(
                title,
                abstract,
            )
            for title, abstract in zip(
                out["title_text"],
                out["abstract_text"],
            )
        ]

        return out

    def _tokenize(
        self,
        source: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        token_rows = []
        field_rows = []
        failures = 0

        for row in source.itertuples(
            index=False
        ):
            rid = str(row.record_id)

            for component in [
                "title",
                "abstract",
            ]:
                text = _normalize_ws(
                    getattr(
                        row,
                        f"{component}_text",
                    )
                )

                try:
                    analyzed = list(
                        self.kiwi.tokenize(
                            text
                        )
                    )
                    field_error = ""

                except Exception as exc:
                    analyzed = []
                    field_error = (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    )
                    failures += 1

                field_rows.append(
                    {
                        "record_id": rid,
                        "component": component,
                        "char_count": len(text),
                        "token_count": len(
                            analyzed
                        ),
                        "analysis_error": (
                            field_error
                        ),
                    }
                )

                for token_seq, tok in enumerate(
                    analyzed,
                    start=1,
                ):
                    token_rows.append(
                        {
                            "record_id": rid,
                            "component": component,
                            "token_seq": (
                                token_seq
                            ),
                            "form": str(
                                tok.form
                            ),
                            "tag": str(
                                tok.tag
                            ),
                            "start": int(
                                tok.start
                            ),
                            "end": int(
                                tok.end
                            ),
                        }
                    )

        if failures != 0:
            raise RuntimeError(
                f"Kiwi field failures: "
                f"{failures}"
            )

        return (
            pd.DataFrame(token_rows),
            pd.DataFrame(field_rows),
        )

    def _generate_candidates(
        self,
        source: pd.DataFrame,
        token_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        token_groups = {
            key: g.copy()
            for key, g in token_df.groupby(
                [
                    "record_id",
                    "component",
                ],
                sort=False,
            )
        }

        candidate_rows = []
        record_rows = []

        for row in source.itertuples(
            index=False
        ):
            rid = str(row.record_id)
            agg = {}

            for component in [
                "title",
                "abstract",
            ]:
                text = _normalize_ws(
                    getattr(
                        row,
                        f"{component}_text",
                    )
                )

                g = token_groups.get(
                    (
                        rid,
                        component,
                    )
                )

                if g is None:
                    raise RuntimeError(
                        f"Missing token group: "
                        f"{rid}/{component}"
                    )

                for occ in _iter_character_spans(
                    text,
                    g,
                ):
                    if _final_refit_accept(
                        occ["tags"],
                        self.full_edge_pairs,
                    ):
                        _add_occurrence(
                            agg,
                            occ,
                            component,
                        )

            record_candidates = (
                _aggregated_candidate_rows(
                    rid,
                    agg,
                )
            )

            candidate_rows.extend(
                record_candidates
            )

            record_rows.append(
                {
                    "record_id": rid,
                    "candidate_primary_rows": (
                        len(
                            record_candidates
                        )
                    ),
                }
            )

        candidates = pd.DataFrame(
            candidate_rows
        )

        if (
            len(candidates)
            and candidates.duplicated(
                [
                    "record_id",
                    "candidate_canonical_key",
                ]
            ).any()
        ):
            raise RuntimeError(
                "Candidate canonical duplicates "
                "detected."
            )

        return (
            candidates,
            pd.DataFrame(record_rows),
        )

    def _build_features(
        self,
        candidates: pd.DataFrame,
    ) -> pd.DataFrame:
        work = candidates.copy()

        for col in [
            "title_occurrence_count",
            "abstract_occurrence_count",
            "total_occurrence_count",
            "min_token_count",
            "max_token_count",
        ]:
            work[col] = pd.to_numeric(
                work[col],
                errors="raise",
            )

        work[
            "candidate_length"
        ] = (
            work[
                "primary_pos_sequence"
            ]
            .astype(str)
            .str.count(r"\+")
            + 1
        ).astype(int)

        work[
            "candidate_length_bin"
        ] = (
            work[
                "candidate_length"
            ]
            .clip(
                upper=GOLD_LENGTH_BIN_CAP
            )
            .astype(int)
        )

        work[
            "phrase_pos_category"
        ] = work[
            "primary_pos_sequence"
        ].map(
            lambda seq:
                _pos_category(
                    tuple(
                        str(seq).split(
                            "+"
                        )
                    )
                )
        )

        work[
            "count_sum_doc_length"
        ] = (
            work.groupby(
                [
                    "record_id",
                    "candidate_length",
                ]
            )[
                "total_occurrence_count"
            ]
            .transform("sum")
        )

        work["tf"] = (
            work[
                "total_occurrence_count"
            ]
            / work[
                "count_sum_doc_length"
            ].replace(
                0,
                np.nan,
            )
        ).fillna(0.0)

        tf_group_sums = (
            work.groupby(
                [
                    "record_id",
                    "candidate_length",
                ]
            )[
                "tf"
            ].sum()
        )

        if len(tf_group_sums):
            max_tf_error = float(
                np.max(
                    np.abs(
                        tf_group_sums.to_numpy(
                            dtype=float
                        )
                        - 1.0
                    )
                )
            )
            if max_tf_error > 1e-12:
                raise RuntimeError(
                    "TF normalization drift."
                )

        work = work.merge(
            self.full_dev_resource[
                [
                    "candidate_canonical_key",
                    "df_fit",
                    "total_occurrences_fit",
                    "sum_count_log_count_fit",
                ]
            ],
            on="candidate_canonical_key",
            how="left",
            validate="many_to_one",
        )

        for col in [
            "df_fit",
            "total_occurrences_fit",
            "sum_count_log_count_fit",
        ]:
            work[col] = pd.to_numeric(
                work[col],
                errors="coerce",
            ).fillna(0)

        work[
            "fit_record_count"
        ] = EXPECTED_DEVELOPMENT_RECORDS

        work["df_fit"] = work[
            "df_fit"
        ].astype(int)

        work["idf_fit"] = (
            np.log(
                (
                    EXPECTED_DEVELOPMENT_RECORDS
                    + 1
                )
                / (
                    work["df_fit"]
                    + 1
                )
            )
            + 1.0
        )

        work["tfidf_raw"] = (
            work["tf"]
            * work["idf_fit"]
        )

        work[
            "tfidf_feature"
        ] = _minmax_within_group(
            work["tfidf_raw"],
            work["record_id"],
        )

        work[
            "df_feature"
        ] = (
            np.log1p(
                work["df_fit"]
            )
            / math.log1p(
                EXPECTED_DEVELOPMENT_RECORDS
            )
        ).clip(
            0.0,
            1.0,
        )

        T = work[
            "total_occurrences_fit"
        ].to_numpy(
            dtype=float
        )

        S = work[
            "sum_count_log_count_fit"
        ].to_numpy(
            dtype=float
        )

        entropy = np.zeros(
            len(work),
            dtype=float,
        )

        positive_mask = T > 0

        entropy[
            positive_mask
        ] = (
            np.log(
                T[positive_mask]
            )
            - (
                S[positive_mask]
                / T[positive_mask]
            )
        )

        entropy = np.clip(
            entropy,
            0.0,
            None,
        )

        work[
            "dispersion_entropy_fit"
        ] = entropy

        work[
            "dispersion_feature"
        ] = (
            work[
                "dispersion_entropy_fit"
            ]
            / math.log(
                EXPECTED_DEVELOPMENT_RECORDS
            )
        ).clip(
            0.0,
            1.0,
        )

        work[
            "dispersion_total_occurrences_fit"
        ] = work[
            "total_occurrences_fit"
        ].astype(int)

        work[
            "dispersion_sum_count_log_count_fit"
        ] = work[
            "sum_count_log_count_fit"
        ].astype(float)

        work[
            "domain_gold_df_fit"
        ] = (
            work[
                "candidate_canonical_key"
            ]
            .map(
                self.domain_map
            )
            .fillna(0)
            .astype(int)
        )

        work[
            "domain_max_gold_df_fit"
        ] = self.max_domain_df

        work[
            "domain_focus_feature"
        ] = (
            np.log1p(
                work[
                    "domain_gold_df_fit"
                ]
            )
            / math.log1p(
                max(
                    self.max_domain_df,
                    1,
                )
            )
        ).clip(
            0.0,
            1.0,
        )

        work[
            "phrase_structural_support_fit"
        ] = (
            work[
                "phrase_pos_category"
            ]
            .map(
                self.category_support
            )
            .fillna(0)
            .astype(int)
        )

        work[
            "phrase_structural_quality"
        ] = (
            work[
                "phrase_pos_category"
            ]
            .map(
                self.category_prior
            )
            .fillna(0.0)
            .astype(float)
        )

        work[
            "phrase_length_support_fit"
        ] = (
            work[
                "candidate_length_bin"
            ]
            .map(
                self.length_support
            )
            .fillna(0)
            .astype(int)
        )

        work[
            "phrase_length_prior"
        ] = (
            work[
                "candidate_length_bin"
            ]
            .map(
                self.length_prior
            )
            .fillna(0.0)
            .astype(float)
        )

        work[
            "phrase_quality_feature"
        ] = (
            PHRASE_STRUCTURAL_WEIGHT
            * work[
                "phrase_structural_quality"
            ]
            + PHRASE_LENGTH_WEIGHT
            * work[
                "phrase_length_prior"
            ]
        ).clip(
            0.0,
            1.0,
        )

        component_matrix = work[
            FIVE_COMPONENT_COLUMNS
        ].to_numpy(
            dtype=float
        )

        if not np.isfinite(
            component_matrix
        ).all():
            raise RuntimeError(
                "Non-finite component value."
            )

        if not (
            (
                component_matrix
                >= -1e-12
            )
            & (
                component_matrix
                <= 1.0 + 1e-12
            )
        ).all():
            raise RuntimeError(
                "Component outside [0,1]."
            )

        return work

    def _score(
        self,
        source: pd.DataFrame,
        features: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        prediction_rows = []
        record_prediction_rows = []

        feature_groups = {
            str(rid): group.copy()
            for rid, group in features.groupby(
                "record_id",
                sort=False,
            )
        }

        for source_row in source.itertuples(
            index=False
        ):
            record_id = str(
                source_row.record_id
            )

            document_group = (
                feature_groups.get(
                    record_id
                )
            )

            if (
                document_group is None
                or len(
                    document_group
                ) == 0
            ):
                selected_rows = []
                top_score = np.nan
                candidate_row_count = 0

            else:
                local = (
                    document_group
                    .sort_values(
                        "candidate_canonical_key",
                        kind="stable",
                    )
                    .reset_index(
                        drop=True
                    )
                )

                feature_matrix = local[
                    FIVE_COMPONENT_COLUMNS
                ].to_numpy(
                    dtype=float
                )

                raw_scores = (
                    feature_matrix
                    @ self.weights
                )

                decision_scores = np.round(
                    raw_scores,
                    self.score_decimals,
                )

                order = np.argsort(
                    -decision_scores,
                    kind="stable",
                )

                top_indices = order[
                    :self.top_k
                ]

                selected_indices = [
                    int(idx)
                    for idx in top_indices
                    if (
                        decision_scores[idx]
                        >= self.threshold
                    )
                ]

                selected_rows = []

                for prediction_rank, idx in enumerate(
                    selected_indices,
                    start=1,
                ):
                    candidate_row = local.iloc[
                        idx
                    ]

                    title_present = (
                        _parse_bool(
                            candidate_row[
                                "title_present"
                            ]
                        )
                    )

                    abstract_present = (
                        _parse_bool(
                            candidate_row[
                                "abstract_present"
                            ]
                        )
                    )

                    if (
                        title_present
                        and abstract_present
                    ):
                        source_field = "TI+AB"
                    elif title_present:
                        source_field = "TI"
                    elif abstract_present:
                        source_field = "AB"
                    else:
                        source_field = ""

                    item = {
                        "record_id": (
                            record_id
                        ),
                        "prediction_rank": (
                            prediction_rank
                        ),
                        "candidate_surface": str(
                            candidate_row[
                                "candidate_surface"
                            ]
                        ),
                        "candidate_canonical": str(
                            candidate_row[
                                "candidate_canonical_key"
                            ]
                        ),
                        "cks_score": float(
                            decision_scores[
                                idx
                            ]
                        ),
                        "raw_cks_score": float(
                            raw_scores[idx]
                        ),
                        "source_field": (
                            source_field
                        ),
                        "source_components": str(
                            candidate_row[
                                "source_components"
                            ]
                        ),
                        "configuration_id": (
                            self.configuration_id
                        ),
                        "minimum_cks_score": (
                            self.threshold
                        ),
                        "reconstruction_status": (
                            "RECONSTRUCTED"
                        ),
                    }

                    prediction_rows.append(
                        item
                    )
                    selected_rows.append(
                        item
                    )

                top_score = (
                    float(
                        decision_scores[
                            order[0]
                        ]
                    )
                    if len(order)
                    else np.nan
                )

                candidate_row_count = len(
                    local
                )

            reconstructed_surfaces = [
                item[
                    "candidate_surface"
                ]
                for item in selected_rows
            ]

            reconstructed_canonical = [
                item[
                    "candidate_canonical"
                ]
                for item in selected_rows
            ]

            if candidate_row_count == 0:
                reconstruction_status = (
                    "NO_ACCEPTED_CANDIDATES"
                )
            elif len(
                selected_rows
            ) == 0:
                reconstruction_status = (
                    "NO_CANDIDATE_ABOVE_THRESHOLD"
                )
            else:
                reconstruction_status = (
                    "RECONSTRUCTED"
                )

            record_prediction_rows.append(
                {
                    "record_id": record_id,
                    "DE_original": str(
                        source_row.DE_original
                    ),
                    "DE_ko_reconstructed": (
                        "; ".join(
                            reconstructed_surfaces
                        )
                    ),
                    "DE_ko_reconstructed_json": (
                        json.dumps(
                            reconstructed_surfaces,
                            ensure_ascii=False,
                        )
                    ),
                    "DE_ko_reconstructed_canonical_json": (
                        json.dumps(
                            reconstructed_canonical,
                            ensure_ascii=False,
                        )
                    ),
                    "reconstructed_keyword_count": (
                        len(
                            reconstructed_surfaces
                        )
                    ),
                    "candidate_rows": (
                        candidate_row_count
                    ),
                    "top_ranked_cks_score": (
                        top_score
                    ),
                    "configuration_id": (
                        self.configuration_id
                    ),
                    "minimum_cks_score": (
                        self.threshold
                    ),
                    "reconstruction_status": (
                        reconstruction_status
                    ),
                    "source_pair_sha256": str(
                        source_row.source_pair_sha256
                    ),
                }
            )

        predictions_long = pd.DataFrame(
            prediction_rows,
            columns=[
                "record_id",
                "prediction_rank",
                "candidate_surface",
                "candidate_canonical",
                "cks_score",
                "raw_cks_score",
                "source_field",
                "source_components",
                "configuration_id",
                "minimum_cks_score",
                "reconstruction_status",
            ],
        )

        reconstruction_by_record = (
            pd.DataFrame(
                record_prediction_rows
            )
        )

        return (
            predictions_long,
            reconstruction_by_record,
        )

    def run_dataframe(
        self,
        input_data: pd.DataFrame,
    ) -> KoreanRunResult:
        source = self._prepare_source(
            input_data
        )

        token_df, field_df = (
            self._tokenize(
                source
            )
        )

        candidates, candidate_summary = (
            self._generate_candidates(
                source,
                token_df,
            )
        )

        features = self._build_features(
            candidates
        )

        predictions_long, by_record = (
            self._score(
                source,
                features,
            )
        )

        return KoreanRunResult(
            source=source,
            token_audit=token_df,
            field_audit=field_df,
            candidate_universe=candidates,
            candidate_record_summary=(
                candidate_summary
            ),
            features=features,
            predictions_long=(
                predictions_long
            ),
            reconstruction_by_record=(
                by_record
            ),
        )


def run_korean_adaptive(
    input_data: pd.DataFrame,
    reference_root: str | Path | None = None,
) -> KoreanRunResult:
    runtime = KoreanAdaptiveCKS(
        reference_root=reference_root
    )
    return runtime.run_dataframe(
        input_data
    )
