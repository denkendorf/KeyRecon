from __future__ import annotations

from ..normalization import normalize_exact_key, normalize_lemma

UNIGRAM_POS = {"NOUN", "PROPN", "ADJ"}
LEFT_POS = {"NOUN", "PROPN", "ADJ", "NUM", "VERB"}
RIGHT_POS = {"NOUN", "PROPN"}
FUNCTION_POS = {"ADP", "AUX", "CCONJ", "DET", "PART", "PRON", "SCONJ"}

class GenericSpacyCandidateGenerator:
    """Conservative multilingual experimental candidate generator.

    It uses the selected spaCy pipeline's tokenization/POS analysis. This is
    intentionally separate from the validated English reference candidate rule.
    """

    status = "experimental"
    min_units = 1
    max_units = 4

    @staticmethod
    def _segments(doc):
        sentences = list(doc.sents) if doc.has_annotation("SENT_START") else [doc[:]]
        for sent in sentences:
            current = []
            for tok in sent:
                if tok.is_space:
                    continue
                if tok.is_punct:
                    if current:
                        yield current
                        current = []
                    continue
                current.append(tok)
            if current:
                yield current

    @staticmethod
    def _accept(span) -> bool:
        if len(span) == 1:
            return span[0].pos_ in UNIGRAM_POS
        if span[0].pos_ in FUNCTION_POS or span[-1].pos_ in FUNCTION_POS:
            return False
        return span[0].pos_ in LEFT_POS and span[-1].pos_ in RIGHT_POS

    def generate_doc(self, doc, source_field: str, record_id: str) -> list[dict]:
        rows = []
        for segment_index, segment in enumerate(self._segments(doc), start=1):
            for n in range(1, min(self.max_units, len(segment)) + 1):
                for i in range(len(segment) - n + 1):
                    toks = segment[i:i+n]
                    if not self._accept(toks):
                        continue
                    start = toks[0].idx
                    end = toks[-1].idx + len(toks[-1].text)
                    surface = doc.text[start:end]
                    lemma = " ".join(normalize_lemma(t.lemma_ or t.text) for t in toks)
                    rows.append({
                        "record_id": str(record_id),
                        "source_field": source_field,
                        "sentence_index": 0,
                        "segment_index": segment_index,
                        "start_char": int(start),
                        "end_char": int(end),
                        "lexical_unit_count": int(n),
                        "candidate_surface": surface,
                        "candidate_exact_key": normalize_exact_key(surface),
                        "candidate_contextual_lemma": normalize_lemma(lemma),
                        "acceptance_rule": "GENERIC_EXPERIMENTAL_POS_BOUNDARY",
                        "left_rule": toks[0].pos_,
                        "right_rule": toks[-1].pos_,
                        "first_pos": toks[0].pos_,
                        "first_tag": toks[0].tag_,
                        "last_pos": toks[-1].pos_,
                        "last_tag": toks[-1].tag_,
                    })
        return rows
