import pandas as pd
from keyrecon.core.features import (
    fit_feature_resources, transform_features, english_reference_structural_quality
)

def test_feature_fitting_bounds():
    candidates = pd.DataFrame({
        "record_id": ["d1", "d1", "d2", "d2", "d3"],
        "canonical_key": ["alpha", "beta", "alpha", "gamma", "alpha"],
        "candidate_length": [1, 1, 1, 1, 1],
        "total_occurrence_count": [2, 1, 1, 3, 1],
    })
    gold = pd.DataFrame({
        "record_id": ["d1", "d2", "d3"],
        "gold_canonical_key": ["alpha", "gamma", "alpha"],
        "gold_length_bin": [1, 1, 1],
    })
    resources = fit_feature_resources(candidates, {"d1", "d2", "d3"}, gold)
    features = transform_features(
        candidates, resources, structural_quality=english_reference_structural_quality
    )
    cols = [
        "tfidf_feature", "df_feature", "dispersion_feature",
        "domain_focus_feature", "phrase_quality_feature",
    ]
    for col in cols:
        assert features[col].between(0.0, 1.0).all()
