import pandas as pd
from keyrecon.config import AdaptiveCKSWeights
from keyrecon.core.scoring import score_features, rank_predictions

def test_score_rounding_and_inclusive_threshold():
    weights = AdaptiveCKSWeights(1.0, 0.0, 0.0, 0.0, 0.0)
    df = pd.DataFrame({
        "record_id": ["d1", "d1"],
        "canonical_key": ["alpha", "beta"],
        "tfidf_feature": [0.44999999999996, 0.45],
        "df_feature": [0.0, 0.0],
        "dispersion_feature": [0.0, 0.0],
        "domain_focus_feature": [0.0, 0.0],
        "phrase_quality_feature": [0.0, 0.0],
    })
    scored = score_features(df, weights)
    assert scored["cks_score"].tolist() == [0.45, 0.45]
    ranked = rank_predictions(scored, 0.45)
    assert ranked["canonical_key"].tolist() == ["alpha", "beta"]

def test_tie_break_is_canonical_key_ascending():
    weights = AdaptiveCKSWeights(1.0, 0.0, 0.0, 0.0, 0.0)
    df = pd.DataFrame({
        "record_id": ["d1", "d1"],
        "canonical_key": ["zeta", "alpha"],
        "tfidf_feature": [0.5, 0.5],
        "df_feature": [0.0, 0.0],
        "dispersion_feature": [0.0, 0.0],
        "domain_focus_feature": [0.0, 0.0],
        "phrase_quality_feature": [0.0, 0.0],
    })
    ranked = rank_predictions(score_features(df, weights), 0.0)
    assert ranked["canonical_key"].tolist() == ["alpha", "zeta"]
