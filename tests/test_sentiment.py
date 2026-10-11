from src.nlp.sentiment import SentimentEngine


def test_fallback_sentiment_sign():
    eng = SentimentEngine(use_transformers=False)
    pos = eng.score("Company beats earnings, raises guidance")
    neg = eng.score("Company misses earnings, warns of bankruptcy")
    assert pos["sentiment"] > 0
    assert neg["sentiment"] < 0
