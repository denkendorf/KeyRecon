# Changelog

## 0.1.0 — 2026-08-19

- Initial KeyRecon package.
- Adds the English `en_reference` profile corresponding to the frozen Adaptive CKS method.
- Adds language-adapter architecture.
- Adds experimental Chinese, Japanese, and Korean spaCy profiles.
- Adds source-aware title/abstract candidate generation.
- Adds leakage-controlled Adaptive CKS feature fitting, deterministic weight search,
  development-only out-of-fold threshold calibration, ranking, evaluation, and provenance manifests.
- No licensed Web of Science source records or document-level research data are bundled.

## 0.2.0

- Add authoritative Korean `ko_reference_v1` frozen at K18.
- Add dedicated Kiwi 0.23.2 Korean runtime verified by K19.
- Promote Korean from experimental generic spaCy to reference.
- Add `keyrecon[korean]` optional dependency.
- Add Korean CLI routing with no per-corpus refitting.
- Preserve the existing English `en_reference` algorithm and profile unchanged.
