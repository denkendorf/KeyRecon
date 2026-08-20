from keyrecon.normalization import normalize_exact_key

def test_normalize_exact_key_dashes_quotes_case_space():
    text = "  Sign\u2013Language  \u2018Test\u2019  "
    assert normalize_exact_key(text) == "sign-language 'test'"
