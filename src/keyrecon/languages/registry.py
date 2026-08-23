from __future__ import annotations

from .base import LanguageAdapter, LanguageSpec

LANGUAGE_SPECS = {
    "en": LanguageSpec(
        code="en",
        label="English",
        model="en_core_web_trf",
        status="reference",
        candidate_profile="en_reference",
        expected_model_version="3.8.0",
        reference_spacy_version="3.8.14",
    ),
    "zh": LanguageSpec(
        code="zh",
        label="Chinese",
        model="zh_core_web_trf",
        status="experimental",
        candidate_profile="generic_experimental",
        expected_model_version="3.8.0",
    ),
    "ja": LanguageSpec(
        code="ja",
        label="Japanese",
        model="ja_core_news_trf",
        status="experimental",
        candidate_profile="generic_experimental",
        expected_model_version="3.8.0",
    ),
    "ko": LanguageSpec(
        code="ko",
        label="Korean",
        model="kiwipiepy",
        status="reference",
        candidate_profile="ko_reference",
        expected_model_version="0.23.2",
        reference_spacy_version=None,
    ),
}

def get_language_spec(language: str) -> LanguageSpec:
    code = language.lower().strip()
    if code not in LANGUAGE_SPECS:
        raise ValueError(f"Unsupported language {language!r}. Choose one of {sorted(LANGUAGE_SPECS)}.")
    return LANGUAGE_SPECS[code]

def make_adapter(
    language: str,
    *,
    model_name: str | None = None,
    strict_reference: bool = False,
) -> LanguageAdapter:
    spec = get_language_spec(language)

    if spec.candidate_profile == "ko_reference":
        raise RuntimeError(
            "The authoritative Korean profile uses the dedicated "
            "`KoreanAdaptiveCKS` runtime. Use `run_korean_adaptive()` "
            "or `keyrecon run ... --lang ko`."
        )

    return LanguageAdapter(
        spec,
        model_name=model_name,
        strict_reference=strict_reference,
    )
