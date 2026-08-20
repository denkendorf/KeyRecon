# Reproducibility

KeyRecon writes a run manifest for CLI reconstructions. The manifest records:

- KeyRecon version;
- selected language and adapter status;
- spaCy and model versions;
- selected Adaptive CKS weights;
- development-calibrated threshold;
- 12-decimal decision-score policy;
- inclusive threshold rule;
- canonicalization mode.

## Exact paper replay

The public repository contains no licensed Web of Science source text. Exact paper-level replay
therefore additionally requires the frozen licensed/private research assets, including the
corpus-derived canonical mapping and the original development/evaluation populations.

The `en_reference` profile stores hashes/fingerprints for those assets so that an authorized
replay can verify that the expected frozen resources are being used.

## Determinism

KeyRecon uses deterministic tie-breaking by canonical key after score ordering. The default
development folds are assigned deterministically from a fixed seed. Users may specify another
seed, but should record it in their study.
