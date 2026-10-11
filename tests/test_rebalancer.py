from src.config import CONFIG
from src.module_a.rebalancer import SentimentRebalancer


def test_weights_sum_to_one_and_tilt():
    reb = SentimentRebalancer(CONFIG)
    sent = {"AAPL": 0.9, "MSFT": -0.9}
    w = reb.compute_weights(sent)
    assert abs(w.sum() - 1.0) < 1e-6
    assert w["AAPL"] > w["MSFT"]
