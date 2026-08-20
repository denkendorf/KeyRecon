from keyrecon.core.weights import generate_weight_candidates

def test_reference_search_space_has_61_weight_vectors():
    weights = generate_weight_candidates()
    assert len(weights) == 61
    assert all(abs(sum(w.as_tuple()) - 1.0) < 1e-12 for w in weights)
