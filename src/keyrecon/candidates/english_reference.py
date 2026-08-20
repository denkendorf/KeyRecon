from __future__ import annotations

import re
from typing import Any

from ..normalization import normalize_exact_key, normalize_lemma

FUNCTION_POS = {"ADP", "AUX", "CCONJ", "DET", "PART", "PRON", "SCONJ"}
FINITE_OR_BASE_TAGS = {"VB", "VBD", "VBP", "VBZ"}
DEFAULT_LEFT_POS = {"ADJ", "NOUN", "NUM", "PROPN"}
DEFAULT_RIGHT_POS = {"NOUN", "PROPN"}
DEFAULT_UNIGRAM_POS = {"ADJ", "NOUN", "PROPN"}
PARTICIPIAL_LEFT_TAGS = {"VBG", "VBN"}
UNIGRAM_VBG_CLAUSAL_DEPS = {"acl", "advcl", "ccomp", "relcl", "xcomp"}

REFERENCE_POLICY = {
    "engine": "spacy_trf",
    "unigram_vbg_relaxed": False,
    "allow_all_unigram_verb": False,
    "allow_multiword_left_verb": False,
    "left_fw_mode": "alpha",
    "alnum_label_mode": "upper",
    "hyphen_rescue_mode": "vb_rb",
    "counterfactual_only": False,
}

def _token_is_alnum_component(tok) -> bool:
    return (
        not tok.is_space
        and not tok.is_punct
        and bool(re.search(r"[A-Za-z0-9]", tok.text))
    )

def _directly_adjacent(left, right) -> bool:
    return left.idx + len(left.text) == right.idx

def _make_unit(doc, token_indices: list[int], hyphenated: bool) -> dict[str, Any]:
    toks = [doc[i] for i in token_indices]
    start = toks[0].idx
    end = toks[-1].idx + len(toks[-1].text)
    surface = doc.text[start:end]
    lexical = [t for t in toks if not t.is_punct and not t.is_space]
    if not lexical:
        raise ValueError(f"Unit has no lexical component: {surface!r}")
    first, last = lexical[0], lexical[-1]
    if hyphenated:
        lemma = "-".join(normalize_lemma(t.lemma_ or t.text) for t in lexical)
    else:
        lemma = normalize_lemma(last.lemma_ or last.text)
    return {
        "start_char": start,
        "end_char": end,
        "surface": surface,
        "lemma": lemma,
        "first_pos": first.pos_,
        "first_tag": first.tag_,
        "first_dep": first.dep_,
        "last_pos": last.pos_,
        "last_tag": last.tag_,
        "last_dep": last.dep_,
        "is_hyphenated_unit": bool(hyphenated),
        "token_indices": token_indices,
    }

def sentence_to_segments(doc, sent):
    indices = list(range(sent.start, sent.end))
    segments, current = [], []
    i = 0
    while i < len(indices):
        ti = indices[i]
        tok = doc[ti]
        if tok.is_space:
            i += 1
            continue
        if _token_is_alnum_component(tok):
            chain = [ti]
            j = i
            while j + 2 < len(indices):
                hy_idx, nx_idx = indices[j + 1], indices[j + 2]
                hy, nx = doc[hy_idx], doc[nx_idx]
                if (
                    hy.text == "-"
                    and _token_is_alnum_component(nx)
                    and _directly_adjacent(doc[indices[j]], hy)
                    and _directly_adjacent(hy, nx)
                ):
                    chain.extend([hy_idx, nx_idx])
                    j += 2
                else:
                    break
            if len(chain) >= 3:
                current.append(_make_unit(doc, chain, hyphenated=True))
                i = j + 1
                continue
        if tok.is_punct:
            if current:
                segments.append(current)
                current = []
            i += 1
            continue
        current.append(_make_unit(doc, [ti], hyphenated=False))
        i += 1
    if current:
        segments.append(current)
    return segments

def _candidate_surface(doc, units):
    return doc.text[units[0]["start_char"]:units[-1]["end_char"]]

def _candidate_lemma(units) -> str:
    return normalize_lemma(" ".join(str(u["lemma"]) for u in units))

def _is_alphanumeric_technical_label(surface: str, mode: str) -> bool:
    surface = str(surface).strip()
    if mode == "none":
        return False
    if mode == "broad":
        return (
            bool(re.fullmatch(r"[A-Za-z0-9]+", surface))
            and bool(re.search(r"[A-Za-z]", surface))
            and bool(re.search(r"\d", surface))
        )
    if mode == "upper":
        return (
            bool(re.fullmatch(r"[A-Z0-9]+", surface))
            and bool(re.search(r"[A-Z]", surface))
            and bool(re.search(r"\d", surface))
        )
    raise ValueError(f"Unknown alnum_label_mode: {mode}")

def _is_whitespace_free_hyphenated_lexical_unit(unit) -> bool:
    surface = str(unit["surface"]).strip()
    return (
        bool(unit.get("is_hyphenated_unit"))
        and bool(re.fullmatch(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+", surface))
        and not bool(re.search(r"\s", surface))
    )

def _hyphenated_exception_ok(unit) -> bool:
    surface = str(unit["surface"])
    return (
        bool(re.fullmatch(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+", surface))
        and not bool(re.search(r"\s", surface))
        and unit["last_tag"] in {"JJ", "VBN"}
    )

def _hyphen_rescue_ok(unit, mode: str) -> bool:
    if mode == "none":
        return False
    if not _is_whitespace_free_hyphenated_lexical_unit(unit):
        return False
    if mode == "broad":
        return True
    if mode == "vb_rb":
        surface = str(unit["surface"]).strip()
        return (
            bool(re.fullmatch(r"[A-Za-z]+(?:-[A-Za-z]+)+", surface))
            and unit["first_tag"] == "VB"
            and unit["last_tag"] == "RB"
        )
    raise ValueError(f"Unknown hyphen_rescue_mode: {mode}")

def _alpha_only_fw_candidate(units) -> bool:
    return (
        2 <= len(units) <= 4
        and all(bool(re.fullmatch(r"[A-Za-z]+", str(u["surface"]))) for u in units)
    )

def evaluate_unigram(unit, policy=REFERENCE_POLICY):
    alnum_mode = policy["alnum_label_mode"]
    hyphen_mode = policy["hyphen_rescue_mode"]
    if _is_alphanumeric_technical_label(unit["surface"], alnum_mode):
        return True, f"UNIGRAM_ALNUM_{alnum_mode.upper()}_TECHNICAL_LABEL_RESCUE", []
    if _hyphen_rescue_ok(unit, hyphen_mode):
        return True, f"UNIGRAM_HYPHEN_{hyphen_mode.upper()}_LEXICAL_RESCUE", []
    if _hyphenated_exception_ok(unit):
        return True, "UNIGRAM_HYPHENATED_LEXICAL_EXCEPTION", []
    if unit["last_pos"] in DEFAULT_UNIGRAM_POS:
        return True, "UNIGRAM_DEFAULT_CONTENT_POS", []
    if policy["allow_all_unigram_verb"] and unit["last_pos"] == "VERB":
        return True, "UNIGRAM_BROAD_VERB_COUNTERFACTUAL", []
    if unit["last_tag"] == "VBG":
        if policy["unigram_vbg_relaxed"]:
            return True, "UNIGRAM_VBG_RELAXED_COUNTERFACTUAL", []
        if unit["last_dep"] not in UNIGRAM_VBG_CLAUSAL_DEPS:
            return True, "UNIGRAM_NOMINALIZED_VBG_STRICT", []
    if unit["last_tag"] in FINITE_OR_BASE_TAGS:
        return False, "", ["UNIGRAM_FINITE_OR_BASE_VERB"]
    if unit["last_pos"] in FUNCTION_POS:
        return False, "", ["UNIGRAM_FUNCTION_POS"]
    return False, "", ["UNIGRAM_POS_NOT_ADMISSIBLE"]

def evaluate_left(unit, policy, all_units):
    if unit["first_pos"] in DEFAULT_LEFT_POS:
        return True, "LEFT_DEFAULT_CONTENT_POS", []
    if unit["first_tag"] in PARTICIPIAL_LEFT_TAGS:
        return True, "LEFT_PARTICIPIAL_VBG_VBN", []
    fw_mode = policy["left_fw_mode"]
    if (
        fw_mode == "broad"
        and unit["first_pos"] == "X"
        and unit["first_tag"] == "FW"
        and bool(re.search(r"[A-Za-z]", str(unit["surface"])))
    ):
        return True, "LEFT_FOREIGN_WORD_FW_BROAD_RESCUE", []
    if (
        fw_mode == "alpha"
        and unit["first_pos"] == "X"
        and unit["first_tag"] == "FW"
        and _alpha_only_fw_candidate(all_units)
    ):
        return True, "LEFT_FOREIGN_WORD_FW_ALPHA_RESCUE", []
    if unit["first_pos"] in FUNCTION_POS:
        return False, "", ["LEFT_FUNCTION_BOUNDARY"]
    if unit["first_pos"] == "VERB":
        if policy["allow_multiword_left_verb"]:
            return True, "LEFT_VERB_BROAD_COUNTERFACTUAL", []
        return False, "", ["LEFT_NONPARTICIPIAL_VERB"]
    return False, "", ["LEFT_POS_NOT_ADMISSIBLE"]

def evaluate_right(unit, policy):
    if unit["last_tag"] in FINITE_OR_BASE_TAGS:
        return False, "", ["RIGHT_FINITE_OR_BASE_VERB"]
    if unit["last_pos"] in DEFAULT_RIGHT_POS:
        return True, "RIGHT_DEFAULT_NOUN_PROPN", []
    if unit["last_tag"] == "VBG":
        return True, "RIGHT_MULTIWORD_VBG_RELAXED", []
    if unit["last_tag"] == "VBN":
        return False, "", ["RIGHT_VBN_NONHYPHENATED"]
    if unit["last_pos"] in FUNCTION_POS:
        return False, "", ["RIGHT_FUNCTION_BOUNDARY"]
    if unit["last_pos"] == "ADJ":
        return False, "", ["RIGHT_ADJ_NONHYPHENATED"]
    if unit["last_pos"] == "ADV":
        return False, "", ["RIGHT_ADV"]
    if unit["last_pos"] == "VERB":
        return False, "", ["RIGHT_OTHER_VERB"]
    return False, "", ["RIGHT_POS_NOT_ADMISSIBLE"]

def evaluate_candidate(units, policy=REFERENCE_POLICY):
    n = len(units)
    if n < 1 or n > 4:
        return False, "", ["SPAN_OUT_OF_RANGE"], "", ""
    if n == 1:
        accepted, rule_id, reasons = evaluate_unigram(units[0], policy)
        return accepted, rule_id, reasons, "UNIGRAM", rule_id if accepted else ""
    internal_units = units[1:-1]
    if any(u["last_tag"] in FINITE_OR_BASE_TAGS for u in internal_units):
        return False, "", ["INTERNAL_FINITE_OR_BASE_VERB"], "", ""
    left_ok, left_rule, left_reasons = evaluate_left(units[0], policy, units)
    right_ok, right_rule, right_reasons = evaluate_right(units[-1], policy)
    reasons = left_reasons + right_reasons
    if not left_ok or not right_ok:
        return False, "", reasons, left_rule, right_rule
    acceptance_rule = f"{left_rule}__{right_rule}"
    return True, acceptance_rule, [], left_rule, right_rule

class EnglishReferenceCandidateGenerator:
    """Candidate generator mirroring the frozen source-aware English evaluator."""

    status = "reference"
    min_units = 1
    max_units = 4

    def generate_doc(self, doc, source_field: str, record_id: str) -> list[dict]:
        rows = []
        for sentence_index, sent in enumerate(doc.sents, start=1):
            for segment_index, segment in enumerate(sentence_to_segments(doc, sent), start=1):
                for n in range(1, min(self.max_units, len(segment)) + 1):
                    for start_i in range(0, len(segment) - n + 1):
                        units = segment[start_i:start_i+n]
                        accepted, rule, reasons, left_rule, right_rule = evaluate_candidate(units)
                        if not accepted:
                            continue
                        surface = _candidate_surface(doc, units)
                        rows.append({
                            "record_id": str(record_id),
                            "source_field": source_field,
                            "sentence_index": sentence_index,
                            "segment_index": segment_index,
                            "start_char": int(units[0]["start_char"]),
                            "end_char": int(units[-1]["end_char"]),
                            "lexical_unit_count": int(n),
                            "candidate_surface": surface,
                            "candidate_exact_key": normalize_exact_key(surface),
                            "candidate_contextual_lemma": _candidate_lemma(units),
                            "acceptance_rule": rule,
                            "left_rule": left_rule,
                            "right_rule": right_rule,
                            "first_pos": units[0]["first_pos"],
                            "first_tag": units[0]["first_tag"],
                            "last_pos": units[-1]["last_pos"],
                            "last_tag": units[-1]["last_tag"],
                        })
        return rows
