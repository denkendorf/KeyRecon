# Multilingual architecture

KeyRecon separates the language-neutral Adaptive CKS core from language-specific candidate logic.

## Current profiles

- `en_reference`: frozen English reference candidate evaluator + English PhraseQuality heuristic.
- `zh_experimental`: `zh_core_web_trf` + conservative generic candidate boundaries.
- `ja_experimental`: `ja_core_news_trf` + conservative generic candidate boundaries.
- `ko_experimental`: `ko_core_news_lg` + conservative generic candidate boundaries.

The experimental adapters should be treated as starting points for validation, not as simple
drop-in replications of the English method.

## Why model substitution is not enough

Candidate generation depends on tokenization, POS/tag inventories, phrase boundaries, and
language-specific morphology. PhraseQuality is also language-sensitive. For this reason,
KeyRecon places those decisions behind a language adapter and keeps the CKS feature/scoring
engine separate.

## Intended extension path

A future adapter can supply another NLP backend while preserving the core API. For example,
a Korean adapter could later use a different morphological analyzer without changing the
Adaptive CKS scoring, calibration, ranking, or provenance modules.
