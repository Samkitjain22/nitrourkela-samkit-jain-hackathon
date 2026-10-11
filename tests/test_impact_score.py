from src.nlp.impact_score import compute_impact

W = {"event": 0.35, "sentiment": 0.20, "source": 0.20, "recency": 0.10, "novelty": 0.15}


def test_impact_range():
    v = compute_impact(0.9, -0.9, 1.0, 1.0, 1.0, W)
    assert 1.0 <= v <= 10.0
    assert v > 8.0


def test_impact_low():
    v = compute_impact(0.1, 0.0, 0.3, 0.0, 0.0, W)
    assert 1.0 <= v <= 3.0
