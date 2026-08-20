from keyrecon.languages.registry import LANGUAGE_SPECS

def test_language_profiles():
    assert LANGUAGE_SPECS["en"].model == "en_core_web_trf"
    assert LANGUAGE_SPECS["en"].status == "reference"
    assert LANGUAGE_SPECS["zh"].model == "zh_core_web_trf"
    assert LANGUAGE_SPECS["ja"].model == "ja_core_news_trf"
    assert LANGUAGE_SPECS["ko"].model == "ko_core_news_lg"
    assert all(LANGUAGE_SPECS[x].status == "experimental" for x in ("zh", "ja", "ko"))
