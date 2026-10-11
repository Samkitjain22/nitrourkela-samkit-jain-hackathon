import math
from datetime import datetime, timezone
from typing import Dict


def recency_score(ts: datetime, half_life_hours: float = 24.0) -> float:
    now = datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    delta_h = max(0.0, (now - ts).total_seconds() / 3600.0)
    return math.exp(-delta_h / half_life_hours)


def source_credibility(source: str, credibility_map: Dict[str, float]) -> float:
    s = (source or "").lower()
    for k, v in credibility_map.items():
        if k != "default" and k in s:
            return v
    return credibility_map.get("default", 0.6)


def compute_impact(event_prior_val: float, sentiment: float,
                   source_cred: float, recency: float,
                   novelty: float, weights: Dict[str, float]) -> float:
    raw = (
        weights["event"] * event_prior_val
        + weights["sentiment"] * abs(sentiment)
        + weights["source"] * source_cred
        + weights["recency"] * recency
        + weights["novelty"] * novelty
    )
    raw = max(0.0, min(1.0, raw))
    return round(1.0 + 9.0 * raw, 2)
