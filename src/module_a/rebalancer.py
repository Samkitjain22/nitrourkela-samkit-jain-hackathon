"""Tactical sentiment-driven index rebalancer (Module A)."""
import pandas as pd


class SentimentRebalancer:
    def __init__(self, cfg):
        self.cfg = cfg["module_a"]

    def compute_weights(self, ticker_sentiment: dict) -> pd.Series:
        tickers = self.cfg["tickers"]
        base = self.cfg["base_weight"]
        w = pd.Series({t: base for t in tickers}, dtype=float)
        for t in tickers:
            s = float(ticker_sentiment.get(t, 0.0))
            w[t] = base * (1.0 + self.cfg["tilt_factor"] * s)
        w = w.clip(lower=self.cfg["min_weight"], upper=self.cfg["max_weight"])
        if w.sum() > 0:
            w = w / w.sum()
        return w

    def decayed_sentiment(self, signals_df: pd.DataFrame,
                          half_life_hours: float = 48.0) -> dict:
        import numpy as np
        if signals_df.empty:
            return {}
        df = signals_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        now = pd.Timestamp.now(tz="UTC")
        age_h = (now - df["timestamp"]).dt.total_seconds() / 3600.0
        df["w"] = np.exp(-age_h / half_life_hours) * df["impact"]
        agg = {}
        for _, r in df.iterrows():
            for t in str(r["tickers"]).strip("[]").replace("'", "").split(","):
                t = t.strip()
                if not t:
                    continue
                agg.setdefault(t, []).append((r["w"], r["sentiment"]))
        out = {}
        for t, pairs in agg.items():
            num = sum(w * s for w, s in pairs)
            den = sum(w for w, _ in pairs) or 1.0
            out[t] = num / den
        return out
