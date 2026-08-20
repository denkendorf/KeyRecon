from keyrecon.config import EN_REFERENCE_WEIGHTS

def test_reference_weights_sum_to_one():
    assert abs(sum(EN_REFERENCE_WEIGHTS.as_tuple()) - 1.0) < 1e-12
