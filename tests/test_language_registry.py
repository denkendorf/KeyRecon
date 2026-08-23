from keyrecon.languages.registry import LANGUAGE_SPECS

def test_language_profiles():
    assert LANGUAGE_SPECS["en"].model == "en_core_web_trf"
    assert LANGUAGE_SPECS["en"].status == "reference"
    assert LANGUAGE_SPECS["zh"].model == "zh_core_web_trf"
    assert LANGUAGE_SPECS["ja"].model == "ja_core_news_trf"
    assert LANGUAGE_SPECS["ko"].model == "kiwipiepy"
    assert LANGUAGE_SPECS["ko"].status == "reference"
    assert LANGUAGE_SPECS["ko"].candidate_profile == "ko_reference"
    assert LANGUAGE_SPECS["ko"].expected_model_version == "0.23.2"
    assert all(LANGUAGE_SPECS[x].status == "experimental" for x in ("zh", "ja"))
