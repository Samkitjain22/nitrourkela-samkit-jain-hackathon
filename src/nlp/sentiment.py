import logging
from typing import Dict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

log = logging.getLogger(__name__)

FINANCE_LEXICON = {
    "beat": 2.0, "beats": 2.0, "miss": -2.0, "misses": -2.0,
    "downgrade": -3.0, "upgrade": 3.0, "bankruptcy": -4.0, "default": -3.5,
    "lawsuit": -2.0, "recall": -2.5, "buyback": 2.5, "dividend": 1.5,
    "guidance": 0.5, "layoffs": -2.0, "acquisition": 1.0, "merger": 1.0,
    "sanction": -2.5, "tariff": -1.5, "restructuring": -1.0, "surge": 2.0,
    "plunge": -2.5, "rally": 2.0, "contracts": -1.5, "expands": 1.5,
}


class SentimentEngine:
    def __init__(self, use_transformers: bool = True,
                 model_name: str = "ProsusAI/finbert"):
        self.use_transformers = use_transformers
        self.pipe = None
        self.vader = SentimentIntensityAnalyzer()
        for w, s in FINANCE_LEXICON.items():
            self.vader.lexicon[w] = s

        if use_transformers:
            try:
                from transformers import pipeline
                self.pipe = pipeline(
                    "sentiment-analysis",
                    model=model_name, tokenizer=model_name,
                    truncation=True, max_length=128,
                )
                log.info("FinBERT loaded.")
            except Exception as e:
                log.warning(f"FinBERT unavailable, using VADER fallback: {e}")
                self.pipe = None

    def score(self, text: str) -> Dict:
        if self.pipe is not None:
            try:
                out = self.pipe(text[:512])[0]
                label = out["label"].lower()
                conf = float(out["score"])
                if label == "positive":
                    s = conf
                elif label == "negative":
                    s = -conf
                else:
                    s = 0.0
                return {"sentiment": round(s, 4), "confidence": round(conf, 4),
                        "model": "finbert"}
            except Exception as e:
                log.warning(f"FinBERT inference failed: {e}")
        v = self.vader.polarity_scores(text)
        return {"sentiment": round(v["compound"], 4),
                "confidence": round(abs(v["compound"]), 4),
                "model": "vader"}
