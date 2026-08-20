from __future__ import annotations

import re
import unicodedata

def normalize_exact_key(text: object) -> str:
    """Frozen English-compatible exact-key normalization."""
    x = unicodedata.normalize("NFKC", str(text))
    x = (
        x.replace("\u2018", "'")
         .replace("\u2019", "'")
         .replace("\u201b", "'")
         .replace("\u2032", "'")
    )
    for dash in ("\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"):
        x = x.replace(dash, "-")
    x = x.casefold()
    return re.sub(r"\s+", " ", x).strip()

def normalize_lemma(text: object) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())
