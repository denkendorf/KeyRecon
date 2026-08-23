from __future__ import annotations

import pytest

from keyrecon import load_korean_reference
from keyrecon.engine import KeyRecon
from keyrecon.languages.registry import LANGUAGE_SPECS


def test_korean_registry_is_authoritative_reference():
    spec = LANGUAGE_SPECS["ko"]

    assert spec.status == "reference"
    assert spec.candidate_profile == "ko_reference"
    assert spec.model == "kiwipiepy"
    assert spec.expected_model_version == "0.23.2"


def test_korean_reference_contract_loads():
    ref = load_korean_reference()

    assert ref.payload["reference_id"] == "ko_reference_v1"
    assert ref.payload["authoritative"] is True
    assert ref.payload["status"] == "FROZEN"
    assert ref.configuration_id == "W034_S45"
    assert ref.threshold == 0.45
    assert ref.score_decimals == 12
    assert ref.top_k == 10
    assert ref.payload["post_core_extension_policy"][
        "authoritative_extensions_enabled"
    ] == []


def test_generic_keyrecon_engine_rejects_korean_reference_path():
    with pytest.raises(RuntimeError):
        KeyRecon(language="ko")
