from datetime import datetime
from typing import List
from pydantic import BaseModel, Field


class Signal(BaseModel):
    id: str
    timestamp: datetime
    source: str
    text: str
    tickers: List[str] = []
    sentiment: float = Field(..., ge=-1.0, le=1.0)
    sentiment_confidence: float = Field(..., ge=0.0, le=1.0)
    event: str
    event_confidence: float = Field(..., ge=0.0, le=1.0)
    impact: float = Field(..., ge=1.0, le=10.0)
    novelty: float = Field(..., ge=0.0, le=1.0)
    recency: float = Field(..., ge=0.0, le=1.0)
    source_credibility: float = Field(..., ge=0.0, le=1.0)
    model_sentiment: str
    model_event: str
