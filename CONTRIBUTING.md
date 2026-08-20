# Contributing

KeyRecon separates the language-neutral Adaptive CKS scoring core from
language-specific candidate adapters.

Please keep these boundaries explicit:

1. Changes to `en_reference` must not silently change the frozen English reference behavior.
2. Experimental language adapters must not be described as validated until independently evaluated.
3. Do not commit licensed Web of Science records, abstracts, held-out labels, or extracted document-level snippets.
4. Add or update tests for every change affecting candidate admissibility, scoring, thresholding, ranking, or provenance.
