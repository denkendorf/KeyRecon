# Scientific boundary

## English reference profile

`en_reference` is the software-facing name for the English reference implementation corresponding
to the frozen Adaptive CKS architecture. Internal research configuration identifiers are not used
as user-facing profile names.

The reference provenance records:

- candidate-rule canonical SHA-256:
  `ea1fdf500e477a228e4c3a77fb7bcd846bebff7163b3ceb24880b09ff5be8b6e`
- candidate-rule file SHA-256:
  `9e62070adf7b00e1a584c9e5e3caf337461389c13a9ec4bd5ca265232d9defd6`
- canonical-mapping file SHA-256:
  `7e12d1a2c74dd34710d64d29b512876ac49ddef1f68a9417e0785dccdf02d61b`
- canonical-mapping semantic fingerprint:
  `a55d5b1ca6bd1eaf09d3ed8bcdd942c0d40471c6d866bfffec50be54304387ef`
- spaCy model: `en_core_web_trf` 3.8.0
- reference spaCy package: 3.8.14

The final scoring profile is:

| component | weight |
|---|---:|
| TF–IDF | 0.3588301778033513 |
| document frequency | 0.1395506716416042 |
| dispersion | 0.1232507796095620 |
| domain focus | 0.2152185351946644 |
| phrase quality | 0.1631498357508180 |

The paper's corpus-specific thresholds, 0.45 and 0.35, are retained as reproduction metadata,
not as universal package defaults. For new corpora, KeyRecon calibrates the absolute threshold
from development-only out-of-fold evidence.

## What is not claimed

- The English weights are not claimed to be universally optimal.
- Chinese/Japanese/Korean profiles are not paper-validated.
- Identity canonicalization is not equivalent to the private frozen paper mapping.
- Operational coverage on records with missing keywords is not the same as reconstruction accuracy.
