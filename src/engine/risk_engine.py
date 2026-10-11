import logging, uuid
from datetime import datetime
from typing import List

import pandas as pd

from src.config import CONFIG
from src.nlp.sentiment import SentimentEngine
from src.nlp.event_classifier import EventClassifier, event_prior
from src.nlp.impact_score import compute_impact, recency_score, source_credibility
from src.nlp.dedup import NoveltyScorer
from src.nlp.entity_extractor import EntityExtractor
from src.engine.schemas import Signal

log = logging.getLogger(__name__)


class RiskEngine:
    def __init__(self):
        cfg = CONFIG
        self.sent = SentimentEngine(
            use_transformers=cfg["nlp"]["use_transformers"],
            model_name=cfg["nlp"]["sentiment_model"],
        )
        self.evt = EventClassifier(use_zero_shot=cfg["nlp"]["use_zero_shot"])
        self.ner = EntityExtractor()
        self.nov = NoveltyScorer(lookback=cfg["impact"]["novelty_lookback"])
        self.weights = cfg["impact"]["weights"]
        self.half_life = cfg["impact"]["recency_half_life_hours"]
        self.cred_map = cfg["sources"]["credibility"]

    def analyze_one(self, source: str, text: str, ts: datetime) -> Signal:
        s = self.sent.score(text)
        e = self.evt.classify(text)
        tickers = self.ner.extract(text)
        novelty = self.nov.novelty(text)
        rec = recency_score(ts, self.half_life)
        cred = source_credibility(source, self.cred_map)
        impact = compute_impact(
            event_prior(e["event"]), s["sentiment"], cred, rec, novelty, self.weights
        )
        return Signal(
            id=str(uuid.uuid4()), timestamp=ts, source=source, text=text,
            tickers=tickers,
            sentiment=s["sentiment"], sentiment_confidence=s["confidence"],
            event=e["event"], event_confidence=e["event_confidence"],
            impact=impact, novelty=novelty, recency=rec, source_credibility=cred,
            model_sentiment=s["model"], model_event=e["method"],
        )

    def analyze_dataframe(self, df: pd.DataFrame) -> List[Signal]:
        out = []
        for _, row in df.iterrows():
            ts = pd.to_datetime(row["timestamp"]).to_pydatetime()
            text = str(row.get("text") or row.get("title") or "")
            src = str(row.get("source", "unknown"))
            out.append(self.analyze_one(src, text, ts))
        return out

    def to_dataframe(self, signals: List[Signal]) -> pd.DataFrame:
        return pd.DataFrame([s.model_dump() for s in signals])
