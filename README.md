# KeyRecon

**KeyRecon** is a Python toolkit for reconstructing missing author keywords using
**Adaptive Composite Keyword Scoring (Adaptive CKS)**.

KeyRecon treats missing author keywords as a metadata-reconstruction problem rather
than as unrestricted keyword extraction. It separates candidate generation,
canonicalization, feature-resource fitting, composite scoring, decision-threshold
calibration, ranking, and provenance.

## Status

KeyRecon v0.2.0 provides validated English and Korean reference profiles with
different runtime and adaptation policies.

- **English (`en_reference`)**: reference implementation of the validated English
  Adaptive CKS architecture. It supports both **fully adaptive** and
  **reference-weight** execution modes.
- **Korean (`ko_reference_v1`)**: authoritative frozen Korean reference profile
  validated with Kiwi 0.23.2. In v0.2.0 it is distributed as a frozen inference
  profile.
- **Chinese (`zh_experimental`)**: experimental language adapter.
- **Japanese (`ja_experimental`)**: experimental language adapter.

The word **reference** in `en_reference` refers to the validated English
language-specific Adaptive CKS architecture and candidate/scoring contract.
It does **not** mean that English weights and thresholds are always frozen.

For English:

- `--mode adaptive` performs full corpus-specific Adaptive CKS fitting;
- `--mode reference` fixes the validated English reference weights while
  re-estimating corpus-specific feature resources and calibrating the threshold.

For Korean v0.2.0:

- `--mode reference` runs the frozen authoritative `ko_reference_v1`;
- `--mode adaptive` is currently retained only as a compatibility alias for the
  same frozen Korean reference and does **not** refit resources, weights, or threshold.

Chinese and Japanese remain experimental. A language model or tokenizer cannot
simply be substituted for another language while assuming that candidate boundaries,
PhraseQuality, weights, or thresholds remain valid.

---

## Why KeyRecon?

Ordinary keyword extraction asks which expressions are salient in a document.

KeyRecon instead assumes that author-keyword metadata are observed for some records
and genuinely missing for others. Observed records can therefore provide
development-only evidence, while records with missing author keywords are the
reconstruction targets.

KeyRecon is designed around the following principles:

- observed author keywords are never overwritten;
- fitting evidence and reconstruction targets are kept separate;
- candidate generation is language-sensitive;
- feature resources are fitted only from permitted development records;
- threshold decisions are deterministic and provenance-aware;
- exact execution settings are recorded for reproducibility.

---

## Installation from PyPI

### English

For the English transformer reference profile:

```bash
python -m pip install "keyrecon[transformers]==0.2.0"
python -m spacy download en_core_web_trf
```

Check the installed English profile:

```bash
keyrecon models
keyrecon profile --lang en
```

### Korean

Install the authoritative Korean reference runtime:

```bash
python -m pip install "keyrecon[korean]==0.2.0"
```

The Korean reference requires:

```text
kiwipiepy==0.23.2
```

Check the installed Korean profile:

```bash
keyrecon models
keyrecon profile --lang ko
```

The Korean installation does not require spaCy.

---

## Install for development

Clone the repository and install the development dependencies:

```bash
python -m pip install -e ".[dev,nlp]"
```

For transformer profiles:

```bash
python -m pip install -e ".[dev,transformers]"
```

Then install the English spaCy pipeline explicitly:

```bash
python -m spacy download en_core_web_trf
```

For Korean development:

```bash
python -m pip install -e ".[dev,korean]"
```

---

## Input format

The basic input format is a CSV containing:

```text
record_id
title
abstract
author_keywords
```

Example:

```csv
record_id,title,abstract,author_keywords
d1,Sign language phonology,This paper studies phonological structure in sign language,sign language;phonology
d2,Syntactic movement,We investigate long-distance movement and locality,movement;locality
d3,Visual language structure,The analysis examines phonological patterning in a visual language,
```

If `author_keywords` is present, records with nonblank author keywords are treated as
observed records and blank rows are treated as reconstruction targets.

Observed author keywords are not overwritten.

---

# English

## English reference profile: `en_reference`

`en_reference` is the validated English Adaptive CKS language profile.

It preserves the validated English candidate-generation and scoring architecture,
while allowing two execution modes:

```text
en_reference + mode=adaptive
    -> fully adaptive CKS

en_reference + mode=reference
    -> reference-weight CKS with corpus-specific calibration
```

These two modes should not be confused.

---

## English fully adaptive mode

Run:

```bash
keyrecon run input.csv predictions.csv --lang en --mode adaptive
```

This is the **fully adaptive English CKS mode**.

Using records with observed author keywords, KeyRecon:

1. generates candidates using the English reference candidate architecture;
2. splits observed records into deterministic out-of-fold development folds;
3. fits feature resources using only the permitted training folds;
4. constructs out-of-fold feature values;
5. searches the prespecified **61 weight vectors × 13 thresholds**;
6. selects the best Adaptive CKS weight vector and threshold from development-only
   out-of-fold evidence;
7. refits the feature resources on all observed records;
8. reconstructs author keywords only for records whose author keywords are missing.

The full adaptive search contains:

```text
61 weight vectors × 13 thresholds = 793 configurations
```

The five Adaptive CKS components are:

- TF–IDF
- document frequency
- dispersion
- domain focus
- phrase quality

---

## English reference-weight mode

Run:

```bash
keyrecon run input.csv predictions.csv --lang en --mode reference
```

In English `reference` mode:

- the validated English reference weights are fixed;
- corpus-specific feature resources are still fitted from observed records;
- the decision threshold is still calibrated from development-only out-of-fold
  evidence.

Therefore English `reference` mode is **not** a completely frozen inference mode.

Conceptually:

```text
English adaptive
    corpus-specific feature resources
    + corpus-specific weights
    + corpus-specific threshold

English reference
    corpus-specific feature resources
    + fixed English reference weights
    + corpus-specific threshold
```

---

## English data sufficiency and calibration

Both English `adaptive` and English `reference` modes require records with
**observed author keywords**, because feature-resource fitting and/or threshold
calibration depend on them.

With the default:

```text
n_folds = 3
```

the current technical minimum is:

```text
3 records with observed author keywords
```

Fewer than 3 observed records cannot be fitted with the default 3-fold setting.

### Important warning

**Three observed records are only the technical minimum required for execution.
They are not a scientifically recommended minimum sample size.**

With only 3 observed records, each held-out fold is trained on only 2 records.
Estimates of document frequency, dispersion, domain focus, PhraseQuality resources,
weights, and/or threshold can therefore be highly unstable.

KeyRecon v0.2.0 does **not** claim that there is a universally validated minimum
number of observed records that guarantees reliable Adaptive CKS calibration.

For small datasets:

- treat fitted results as exploratory;
- inspect the reported number of observed fitting records;
- inspect out-of-fold evaluation metrics;
- inspect selected weights and threshold;
- do not interpret successful execution as a guarantee of reconstruction accuracy.

A dataset-specific stability or sensitivity analysis is recommended when calibration
reliability is substantively important.

---

## English Python API

```python
import pandas as pd
from keyrecon import KeyRecon

records = pd.read_csv(
    "input.csv",
    keep_default_na=False,
)

model = KeyRecon(
    language="en",
    mode="adaptive",
)

predictions = model.fit_reconstruct_missing(records)

print(model.fit_summary_)

print(
    predictions[
        [
            "record_id",
            "canonical_key",
            "cks_score",
            "rank",
        ]
    ]
)
```

For English reference-weight mode:

```python
model = KeyRecon(
    language="en",
    mode="reference",
)
```

---

# Korean

## Korean authoritative reference: `ko_reference_v1`

KeyRecon v0.2.0 includes the validated Korean authoritative reference:

```text
ko_reference_v1
```

The Korean reference uses:

```text
Analyzer              : Kiwi
Required Kiwi version : 0.23.2
Configuration         : W034_S45
Threshold             : 0.45
Decision rounding     : 12 decimals
Threshold rule        : rounded_score >= 0.45
Top-k                 : 10
```

The validated Korean Adaptive CKS weights are:

```text
TF-IDF        0.4152282333920128
DF            0.1598348330172827
Dispersion    0.0805437661577038
DomainFocus   0.1426596555299836
PhraseQuality 0.2017335119030170
```

These values were selected during the Korean Adaptive CKS development and validation
process and are distributed in v0.2.0 as a frozen authoritative reference.

---

## Korean reference mode

Run:

```bash
keyrecon run input.csv predictions.csv --lang ko --mode reference
```

In v0.2.0, Korean reference mode uses the frozen:

```text
candidate contract
canonicalization contract
feature resources
weights
threshold
ranking contract
```

from `ko_reference_v1`.

It does **not** estimate new feature resources, weights, or threshold from the
user's corpus.

---

## Korean `adaptive` mode in v0.2.0

The following command is accepted:

```bash
keyrecon run input.csv predictions.csv --lang ko --mode adaptive
```

However, in **KeyRecon v0.2.0**, this is a compatibility alias for:

```bash
keyrecon run input.csv predictions.csv --lang ko --mode reference
```

It therefore runs the same frozen `ko_reference_v1`.

It does **not** currently perform corpus-specific Korean Adaptive CKS fitting.

In particular, it does not:

- refit Korean DF/IDF resources;
- refit Korean dispersion resources;
- refit Korean DomainFocus resources;
- refit Korean PhraseQuality priors;
- search new Korean weights;
- calibrate a new Korean threshold.

A future full-adaptive Korean implementation would require these resources to be
re-estimated from records with observed Korean author keywords using leakage-controlled
out-of-fold fitting.

---

## Korean data requirements

Because `ko_reference_v1` is a frozen inference profile, it does not require a
minimum number of observed author-keyword records for per-corpus calibration.

In principle, one or more target records can be processed.

However:

**the ability to run inference on a small number of records does not imply that
reconstruction accuracy is guaranteed.**

Korean reconstruction quality can still depend on:

- availability and quality of title and abstract;
- similarity between the user's domain and the validation domain;
- language composition;
- candidate coverage;
- terminology distribution;
- the scope of the frozen Korean validation.

Therefore `ko_reference_v1` should be understood as a validated frozen reference,
not as a guarantee of identical accuracy in every external corpus.

---

## English and Korean mode comparison

| language/profile | mode | feature resources | weights | threshold | status |
|---|---|---|---|---|---|
| `en_reference` | `adaptive` | corpus-specific | corpus-specific | corpus-specific | **fully adaptive** |
| `en_reference` | `reference` | corpus-specific | fixed English reference weights | corpus-specific | reference-weight |
| `ko_reference_v1` | `reference` | frozen Korean reference | frozen | frozen at 0.45 | authoritative reference |
| `ko_reference_v1` | `adaptive` | frozen Korean reference | frozen | frozen at 0.45 | v0.2.0 compatibility alias |

This distinction is important:

> `en_reference` is the validated English **language profile**, not the name of a
> completely frozen English inference configuration.

The English profile can be used in fully adaptive mode.

By contrast, Korean v0.2.0 currently distributes the validated Korean Adaptive CKS
result as a fully frozen reference profile.

---

## Language profiles

| code | profile | runtime / NLP model | status |
|---|---|---|---|
| `en` | `en_reference` | `en_core_web_trf` | reference; adaptive and reference-weight modes |
| `ko` | `ko_reference_v1` | `kiwipiepy==0.23.2` | authoritative frozen reference |
| `zh` | `zh_experimental` | `zh_core_web_trf` | experimental |
| `ja` | `ja_experimental` | `ja_core_news_trf` | experimental |

Language selection is explicit for reproducibility.

Automatic language detection is intentionally not enabled in v0.2.0.

---

## Scientific boundary

### English

The English reference profile preserves the validated study architecture:

- title and abstract are parsed independently before candidate union;
- the reference candidate span is 1–4 lexical units;
- source provenance is retained;
- no title score bonus is used;
- the five score components are TF–IDF, document frequency, dispersion,
  domain focus, and phrase quality;
- the operational decision score is rounded to 12 decimals;
- eligibility uses an inclusive threshold;
- ranking is score-descending with canonical-key ascending as the deterministic
  tie-break;
- at most 10 predictions are returned;
- no below-threshold fallback is used.

In `--mode adaptive`, weights and threshold are selected from development-only
out-of-fold evidence while preserving this validated architecture.

### Korean

The Korean authoritative reference preserves the validated Korean-specific
architecture, including:

- Kiwi 0.23.2;
- the frozen Korean candidate rule;
- Korean canonicalization;
- Korean TF–IDF, DF, and dispersion resource definitions;
- Korean DomainFocus;
- Korean PhraseQuality based on validated Korean priors;
- the validated W034_S45 weight configuration;
- threshold 0.45;
- 12-decimal decision-score rounding;
- inclusive thresholding;
- deterministic ranking;
- top-k 10.

The Korean v0.2.0 runtime does not perform post-release retuning.

---

## Reproducibility boundary

The public package does **not** bundle licensed or private research corpora,
including licensed Web of Science records, held-out gold labels, or other source
records that cannot be redistributed.

Therefore exact paper-level replay may require the private or licensed frozen
research assets.

The public package provides:

- the executable Adaptive CKS implementation;
- validated reference contracts;
- runtime resources that can legally be distributed;
- package and release provenance;
- deterministic scoring and ranking rules.

See:

```text
docs/scientific_boundary.md
docs/reproducibility.md
```

---

## Multilingual design

The Adaptive CKS scoring core is separated from language-specific candidate and
feature logic.

Replacing the NLP model alone is not sufficient evidence of multilingual validity.

The following are language-sensitive:

- tokenization;
- candidate boundaries;
- POS behavior;
- canonicalization;
- author-keyword length distribution;
- PhraseQuality;
- corpus-resource fitting;
- optimal weights;
- decision thresholds.

For this reason, language profiles are validated separately.

See:

```text
docs/multilingual.md
```

---

## Output

A KeyRecon run writes:

- a long-form prediction CSV;
- a JSON run manifest containing the execution and provenance information.

Depending on the execution mode, the manifest may include:

- language/profile;
- selected mode;
- observed fitting-record count;
- selected weights;
- selected threshold;
- out-of-fold metrics;
- runtime/model information;
- decision-score and ranking policy.

---

## Inspect installed profiles

Use:

```bash
keyrecon models
```

English:

```bash
keyrecon profile --lang en
```

Korean:

```bash
keyrecon profile --lang ko
```

---

## Version

Current public release:

```text
KeyRecon 0.2.0
Git tag: v0.2.0
```

---

## Citation

See:

```text
CITATION.cff
```

---

## License

MIT.
