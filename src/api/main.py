"""FastAPI service. Run: uvicorn src.api.main:app --reload --port 8000"""
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.config import CONFIG
from src.engine.risk_engine import RiskEngine
from src.engine.schemas import Signal

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

app = FastAPI(title="AI/NLP Risk Engine", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

engine = RiskEngine()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ts": datetime.now(timezone.utc).isoformat(),
        "sentiment_backend": "finbert" if engine.sent.pipe else "vader",
        "event_backend": "zero-shot" if engine.evt.pipe else "keyword",
    }


@app.post("/analyze", response_model=Signal)
def analyze(payload: dict):
    text = payload.get("text", "")
    if not text:
        raise HTTPException(400, "text required")
    src = payload.get("source", "manual")
    return engine.analyze_one(src, text, datetime.now(timezone.utc))


@app.get("/signals")
def signals(limit: int = 100):
    path = Path(CONFIG["paths"]["signals_csv"])
    if not path.exists():
        return []
    df = pd.read_csv(path).tail(limit)
    return df.to_dict(orient="records")


@app.get("/signals/{ticker}")
def signals_by_ticker(ticker: str, limit: int = 100):
    path = Path(CONFIG["paths"]["signals_csv"])
    if not path.exists():
        return []
    df = pd.read_csv(path)
    df = df[df["tickers"].astype(str).str.contains(ticker.upper(), na=False)]
    return df.tail(limit).to_dict(orient="records")
