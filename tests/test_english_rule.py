from keyrecon.candidates.english_reference import evaluate_candidate

def unit(surface, pos="NOUN", tag="NN", dep="compound", hyphen=False):
    return {
        "surface": surface,
        "first_pos": pos,
        "first_tag": tag,
        "first_dep": dep,
        "last_pos": pos,
        "last_tag": tag,
        "last_dep": dep,
        "is_hyphenated_unit": hyphen,
    }

def test_default_noun_unigram_is_accepted():
    ok, rule, reasons, _, _ = evaluate_candidate([unit("phonology")])
    assert ok
    assert rule == "UNIGRAM_DEFAULT_CONTENT_POS"
    assert not reasons

def test_upper_alphanumeric_technical_label_rescue():
    ok, rule, *_ = evaluate_candidate([unit("N400", pos="X", tag="FW")])
    assert ok
    assert "ALNUM_UPPER" in rule

def test_lowercase_yearlike_label_not_rescued():
    ok, *_ = evaluate_candidate([unit("2004a", pos="X", tag="FW")])
    assert not ok

def test_internal_finite_verb_is_rejected():
    units = [
        unit("language", pos="NOUN", tag="NN"),
        unit("is", pos="AUX", tag="VBZ"),
        unit("structure", pos="NOUN", tag="NN"),
    ]
    ok, _, reasons, _, _ = evaluate_candidate(units)
    assert not ok
    assert "INTERNAL_FINITE_OR_BASE_VERB" in reasons
