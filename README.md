# KeyRecon

**KeyRecon** is a Python toolkit for reconstructing missing author keywords using
**Adaptive Composite Keyword Scoring (Adaptive CKS)**.

KeyRecon treats missing author keywords as a metadata-reconstruction problem rather
than as unrestricted keyword extraction. It separates candidate generation, canonicalization,
feature fitting, composite scoring, decision-threshold calibration, ranking, and provenance.

## Status

- **English (`en_reference`)**: reference implementation of the frozen Adaptive CKS architecture.
- **Chinese (`zh_experimental`)**: experimental language adapter.
- **Japanese (`ja_experimental`)**: experimental language adapter.
- **Korean (`ko_experimental`)**: experimental language adapter.

The multilingual adapters are deliberately marked **experimental**. The Adaptive CKS paper
evaluates the English reference architecture; it does not establish that the same candidate
policy, weights, or thresholds are optimal in Chinese, Japanese, or Korean.

## Why KeyRecon?

Ordinary keyword extraction asks which expressions are salient in a document.
KeyRecon instead assumes that author-keyword metadata are observed for some records and
genuinely missing for others. Observed records can therefore supply development-only evidence,
while missing records are reconstruction targets. Observed author keywords are never overwritten.

## Install for development

```bash
python -m pip install -e ".[dev,nlp]"
```

For transformer profiles:

```bash
python -m pip install -e ".[dev,transformers]"
```

Then install the spaCy pipeline explicitly:

```bash
python -m spacy download en_core_web_trf
```

Available KeyRecon profiles can be inspected with:

```bash
keyrecon models
keyrecon setup --lang en
keyrecon profile --lang en
```

## Minimal CSV workflow

Input:

```csv
record_id,title,abstract,author_keywords
d1,Sign language phonology,This paper studies phonological structure in sign language,sign language;phonology
d2,Syntactic movement,We investigate long-distance movement and locality,movement;locality
d3,Visual language structure,The analysis examines phonological patterning in a visual language,
```

Run:

```bash
keyrecon run input.csv predictions.csv --lang en --mode adaptive
```

`--mode adaptive` searches the prespecified 61-weight × 13-threshold Adaptive CKS space
using development-only out-of-fold evidence from records with observed author keywords.

For an English reference-weight run with development-only threshold calibration:

```bash
keyrecon run input.csv predictions.csv --lang en --mode reference
```

The command writes a long-form prediction CSV and a JSON run manifest.

## Python API

```python
import pandas as pd
from keyrecon import KeyRecon

records = pd.read_csv("input.csv", keep_default_na=False)

model = KeyRecon(language="en", mode="adaptive")
predictions = model.fit_reconstruct_missing(records)

print(model.fit_summary_)
print(predictions[["record_id", "canonical_key", "cks_score", "rank"]])
```

## Language profiles

| code | profile | spaCy pipeline | status |
|---|---|---|---|
| `en` | `en_reference` | `en_core_web_trf` | reference |
| `zh` | `zh_experimental` | `zh_core_web_trf` | experimental |
| `ja` | `ja_experimental` | `ja_core_news_trf` | experimental |
| `ko` | `ko_experimental` | `ko_core_news_lg` | experimental |

Language selection is explicit for reproducibility. Automatic language detection is intentionally
not enabled in v0.1.0.

## Scientific boundary

The English reference profile preserves the study's key invariants:

- title and abstract are parsed independently before candidate union;
- the reference candidate span is 1–4 lexical units;
- source provenance is retained and no title score bonus is used;
- the five score components are TF–IDF, document frequency, dispersion, domain focus,
  and phrase quality;
- the operational decision score is rounded to 12 decimals;
- eligibility is inclusive (`rounded_score >= threshold`);
- ranking is score-descending with canonical-key ascending as deterministic tie-break;
- at most 10 predictions are returned;
- no below-threshold fallback is used.

The public package does **not** bundle licensed Web of Science records, abstracts, held-out gold
labels, or the full corpus-derived canonical mapping. Therefore exact paper-level replay also
requires the private/licensed frozen research assets. The package provides the algorithmic
reference implementation and provenance hashes for those external frozen assets.

See `docs/scientific_boundary.md` and `docs/reproducibility.md`.

## Multilingual design

The Adaptive CKS scoring core is separated from the language adapter. Replacing the NLP model
alone is not treated as sufficient evidence of multilingual validity: tokenization, candidate
boundaries, POS behavior, and PhraseQuality are language-sensitive.

See `docs/multilingual.md`.

## Citation

See `CITATION.cff`.

## License

MIT.

## Korean authoritative reference (0.2.0)

Install the frozen Korean reference runtime:

```bash
pip install 'keyrecon[korean]'
```

Run the authoritative Korean reference:

```bash
keyrecon run input.csv output.csv --lang ko --mode reference
```

`--mode adaptive` is retained for Korean as a compatibility alias for the same frozen `ko_reference_v1`; it does not refit weights or threshold.

The existing English `en_reference` profile is unchanged.
