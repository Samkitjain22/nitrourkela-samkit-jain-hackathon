"""End-to-end batch pipeline."""
import json
import logging
from pathlib import Path

import pandas as pd

from src.config import CONFIG
from src.engine.risk_engine import RiskEngine

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("pipeline")


def main():
    news = pd.read_csv(CONFIG["paths"]["news_csv"])
    tweets = pd.read_csv(CONFIG["paths"]["tweets_csv"])
    df = pd.concat([news, tweets], ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    log.info(f"Ingested {len(df)} items from news + tweets")
    engine = RiskEngine()
    signals = engine.analyze_dataframe(df)
    out = engine.to_dataframe(signals)

    Path(CONFIG["paths"]["signals_csv"]).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(CONFIG["paths"]["signals_csv"], index=False)
    with open(CONFIG["paths"]["signals_json"], "w") as f:
        json.dump(out.to_dict(orient="records"), f, default=str, indent=2)

    log.info(f"Wrote {len(out)} signals -> {CONFIG['paths']['signals_csv']}")
    log.info(f"Top 5 by impact:\n{out.nlargest(5, 'impact')[['timestamp','source','event','sentiment','impact']].to_string(index=False)}")


if __name__ == "__main__":
    main()
