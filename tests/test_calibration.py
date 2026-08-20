import pandas as pd
from keyrecon.config import AdaptiveCKSWeights
from keyrecon.core.calibration import calibrate_threshold

def test_calibration_prefers_best_f1_then_lower_threshold():
    # One feature only: alpha scores 0.8, beta scores 0.2.
    features = pd.DataFrame({
        "record_id": ["d1", "d1", "d2", "d2"],
        "canonical_key": ["alpha", "beta", "alpha", "beta"],
        "tfidf_feature": [0.8, 0.2, 0.8, 0.2],
        "df_feature": [0,0,0,0],
        "dispersion_feature": [0,0,0,0],
        "domain_focus_feature": [0,0,0,0],
        "phrase_quality_feature": [0,0,0,0],
    })
    gold = pd.DataFrame({
        "record_id": ["d1", "d2"],
        "gold_canonical_key": ["alpha", "alpha"],
    })
    weights = AdaptiveCKSWeights(1,0,0,0,0)
    result = calibrate_threshold(
        features, gold, weights=weights, thresholds=[0.0, 0.5, 0.8],
        fold_lookup={"d1":0, "d2":1},
    )
    # 0.5 and 0.8 both yield only alpha; tie-break chooses lower threshold.
    assert result.threshold == 0.5
    assert result.metrics["f1_top10"] == 1.0
