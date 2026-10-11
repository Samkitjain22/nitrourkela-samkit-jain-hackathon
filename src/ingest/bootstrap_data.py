"""
Bootstrap all synthetic + cached data so the demo runs 100% offline on a MacBook.
Run: python -m src.ingest.bootstrap_data
"""
from pathlib import Path
import random
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

NEWS = [
    ("2026-02-10T09:15:00Z","reuters",  "Fed holds rates steady, signals two cuts in 2026"),
    ("2026-02-10T11:40:00Z","bloomberg","JPMorgan beats Q4 earnings on strong trading revenue"),
    ("2026-02-11T07:05:00Z","reuters",  "Oil spikes 8% as Middle East tensions escalate"),
    ("2026-02-11T13:22:00Z","ft",       "Apple unveils Vision Pro 3 with on-device AI assistant"),
    ("2026-02-11T15:47:00Z","reuters",  "Moody's downgrades legacy Credit Suisse bonds to junk"),
    ("2026-02-12T08:00:00Z","bloomberg","Microsoft to acquire AI startup for $12B in all-cash deal"),
    ("2026-02-12T10:30:00Z","reuters",  "Tesla recalls 200,000 vehicles over software bug"),
    ("2026-02-12T14:15:00Z","ft",       "Nvidia unveils next-generation Blackwell Ultra chips"),
    ("2026-02-13T12:30:00Z","bloomberg","US CPI comes in hotter than expected at 3.4%"),
    ("2026-02-13T09:10:00Z","reuters",  "Boeing faces new FAA probe after engine incident"),
    ("2026-02-14T08:20:00Z","wsj",      "Amazon to spin off AWS logistics unit next year"),
    ("2026-02-14T06:45:00Z","reuters",  "China stimulus package lifts Asian markets"),
    ("2026-02-15T16:00:00Z","bloomberg","Regional bank SVB Financial files for bankruptcy"),
    ("2026-02-15T11:00:00Z","reuters",  "Google antitrust ruling could force Chrome sale"),
    ("2026-02-16T09:30:00Z","ft",       "Johnson & Johnson wins FDA approval for new cancer drug"),
    ("2026-02-16T14:00:00Z","reuters",  "Walmart raises full-year guidance on strong consumer"),
    ("2026-02-17T07:45:00Z","bloomberg","Visa and Mastercard face new interchange fee rules"),
    ("2026-02-17T13:10:00Z","reuters",  "Exxon Mobil announces $60B share buyback"),
    ("2026-02-18T10:20:00Z","wsj",      "Procter & Gamble warns of margin pressure from tariffs"),
    ("2026-02-18T15:30:00Z","bloomberg","UnitedHealth CEO steps down amid investigation"),
    ("2026-02-19T08:00:00Z","reuters",  "Home Depot cuts outlook on weak housing market"),
    ("2026-02-19T12:45:00Z","ft",       "Bank of America raises dividend 15%"),
    ("2026-02-20T09:15:00Z","reuters",  "Meta fined 1.3B euros by EU over data transfers"),
    ("2026-02-20T13:40:00Z","bloomberg","Russia-Ukraine ceasefire talks stall, gas prices surge"),
    ("2026-02-21T07:20:00Z","reuters",  "Taiwan strait tensions rise after naval incident"),
    ("2026-02-21T11:50:00Z","bloomberg","OPEC+ cuts production by 1M barrels per day"),
    ("2026-02-22T13:30:00Z","reuters",  "US jobs report blows past expectations, wage growth accelerates"),
    ("2026-02-22T15:55:00Z","ft",       "Moody's downgrades commercial real estate sector outlook"),
    ("2026-02-23T10:05:00Z","bloomberg","Ford and GM warn of EV demand slowdown"),
    ("2026-02-23T14:25:00Z","reuters",  "Biotech IPO window reopens with three listings"),
    ("2026-02-24T09:00:00Z","bloomberg","Apple supplier warns of chip shortage in Q2"),
    ("2026-02-24T12:10:00Z","reuters",  "Federal Reserve officials split on pace of rate cuts"),
    ("2026-02-25T08:40:00Z","ft",       "Nvidia beats earnings, guidance above street"),
    ("2026-02-25T14:00:00Z","bloomberg","JPMorgan raises net interest income outlook"),
    ("2026-02-26T09:25:00Z","reuters",  "US 10-year yield jumps 15bp after CPI surprise"),
    ("2026-02-26T11:15:00Z","bloomberg","S&P downgrades regional bank sector to negative"),
    ("2026-02-27T10:50:00Z","reuters",  "Pfizer announces job cuts amid restructuring"),
    ("2026-02-27T13:30:00Z","ft",       "US Treasury warns of debt ceiling deadline in June"),
    ("2026-02-28T08:00:00Z","reuters",  "Gold hits all-time high on geopolitical risk"),
    ("2026-02-28T12:00:00Z","bloomberg","Brent crude above $95 on OPEC+ cuts"),
    ("2026-03-01T09:00:00Z","reuters",  "China property crisis deepens with Evergrande default"),
    ("2026-03-01T14:20:00Z","bloomberg","US regional bank deposits stabilize after outflows"),
    ("2026-03-02T08:30:00Z","ft",       "European banks rally on rate-cut hopes"),
    ("2026-03-02T12:45:00Z","reuters",  "Boeing secures $20B order from Middle East airline"),
    ("2026-03-03T10:10:00Z","bloomberg","Nvidia announces $10B accelerated buyback"),
    ("2026-03-03T13:50:00Z","reuters",  "US manufacturing PMI contracts for third month"),
]

TWEETS = [
    ("2026-02-10T09:18:00Z","twitter","$JPM beats earnings, best quarter yet! 🚀"),
    ("2026-02-10T11:45:00Z","twitter","Fed pivot incoming. $SPY to the moon"),
    ("2026-02-11T07:10:00Z","x",      "Oil +8% and $XOM barely moves, something's off"),
    ("2026-02-11T13:25:00Z","twitter","$AAPL Vision Pro 3 is a game changer"),
    ("2026-02-11T15:50:00Z","x",      "Credit Suisse downgraded again 😬 $CS"),
    ("2026-02-12T08:05:00Z","twitter","$MSFT buying another AI startup, $12B!!"),
    ("2026-02-12T10:35:00Z","x",      "$TSLA recall again, this is getting old"),
    ("2026-02-12T14:20:00Z","twitter","$NVDA Blackwell Ultra is insane. Buy the dip."),
    ("2026-02-13T12:35:00Z","twitter","CPI hot 3.4%, rates staying higher for longer 📉"),
    ("2026-02-13T09:15:00Z","x",      "$BA FAA probe, another hit for Boeing"),
    ("2026-02-14T08:25:00Z","twitter","$AMZN spinning off AWS logistics? Huge value unlock"),
    ("2026-02-14T06:50:00Z","x",      "China stimulus, $FXI ripping"),
    ("2026-02-15T16:05:00Z","twitter","$SIVB bankrupt, contagion fears rising"),
    ("2026-02-15T11:05:00Z","x",      "$GOOGL antitrust could force Chrome sale, wild"),
    ("2026-02-16T09:35:00Z","twitter","$JNJ new cancer drug approved 🎉"),
    ("2026-02-16T14:05:00Z","x",      "$WMT guidance raise, consumer is strong"),
    ("2026-02-17T07:50:00Z","twitter","$V $MA interchange rules hurting margins"),
    ("2026-02-17T13:15:00Z","x",      "$XOM $60B buyback, management confident"),
    ("2026-02-18T10:25:00Z","twitter","$PG tariff margin pressure, ouch"),
    ("2026-02-18T15:35:00Z","x",      "$UNH CEO out, stock tanking"),
    ("2026-02-19T08:05:00Z","twitter","$HD cuts outlook, housing market weak"),
    ("2026-02-19T12:50:00Z","x",      "$BAC dividend raise 15%, solid"),
    ("2026-02-20T09:20:00Z","twitter","$META EU fine 1.3B, regulators gone wild"),
    ("2026-02-20T13:45:00Z","x",      "Russia Ukraine talks stall, energy prices rising"),
    ("2026-02-21T07:25:00Z","twitter","Taiwan tensions rising, semis at risk $NVDA $TSM"),
    ("2026-02-21T11:55:00Z","x",      "OPEC cuts 1M bpd, oil to $100 soon"),
    ("2026-02-22T13:35:00Z","twitter","Jobs report blowout, Fed cuts delayed 📉"),
    ("2026-02-22T16:00:00Z","x",      "CRE outlook downgraded, regional banks wobble"),
    ("2026-02-23T10:10:00Z","twitter","$F $GM EV demand slowing, tough tape"),
    ("2026-02-23T14:30:00Z","x",      "Biotech IPO window reopening, risk-on"),
    ("2026-02-24T09:05:00Z","twitter","Apple supplier warns chip shortage $AAPL"),
    ("2026-02-24T12:15:00Z","x",      "Fed officials split on cuts, uncertainty high"),
    ("2026-02-25T08:45:00Z","twitter","$NVDA earnings beat, guidance raised 🚀🚀"),
    ("2026-02-25T14:05:00Z","x",      "$JPM NII outlook raised, banks looking strong"),
    ("2026-02-26T09:30:00Z","twitter","10y yield +15bp, duration getting crushed"),
    ("2026-02-26T11:20:00Z","x",      "S&P downgrades regional banks to negative 😬"),
    ("2026-02-27T10:55:00Z","twitter","$PFE job cuts, restructuring mode"),
    ("2026-02-27T13:35:00Z","x",      "Debt ceiling deadline June, buckle up"),
    ("2026-02-28T08:05:00Z","twitter","Gold ATH, geopolitical premium rising"),
    ("2026-02-28T12:05:00Z","x",      "Brent above $95, energy sector outperforming"),
    ("2026-03-01T09:05:00Z","twitter","Evergrande default, China property crisis deepens"),
    ("2026-03-01T14:25:00Z","x",      "US regional bank deposits stabilizing, relief rally"),
    ("2026-03-02T08:35:00Z","twitter","European banks rallying on rate cut hopes $EUFN"),
    ("2026-03-02T12:50:00Z","x",      "$BA secures $20B order, big win"),
    ("2026-03-03T10:15:00Z","twitter","$NVDA $10B buyback, another leg up"),
    ("2026-03-03T13:55:00Z","x",      "US manufacturing PMI contraction, recession watch"),
]

def build_news_df():
    rows = [{"timestamp": t, "source": s, "title": title, "text": title} for t, s, title in NEWS]
    return pd.DataFrame(rows)

def build_tweets_df():
    rows = [{"timestamp": t, "source": s, "title": txt, "text": txt} for t, s, txt in TWEETS]
    return pd.DataFrame(rows)

def build_prices_df(tickers, days=365):
    try:
        import yfinance as yf
        df = yf.download(tickers, period=f"{days}d", interval="1d",
                         progress=False, auto_adjust=True)["Close"]
        if df is not None and not df.empty:
            df = df.dropna(how="all").ffill().dropna()
            print(f"[prices] yfinance OK: {df.shape}")
            return df
    except Exception as e:
        print(f"[prices] yfinance failed ({e}), using synthetic GBM")
    dates = pd.date_range(end=pd.Timestamp.utcnow().normalize(), periods=days, freq="B")
    n = len(tickers)
    mu = np.random.uniform(-0.0002, 0.0008, n)
    sigma = np.random.uniform(0.012, 0.028, n)
    shocks = np.random.normal(0, 1, (days, n))
    log_ret = mu + sigma * shocks
    prices = 100 * np.exp(np.cumsum(log_ret, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)

RATINGS = ["AAA","AA","A","BBB","BB","B","CCC"]
SECTORS = ["Tech","Financials","Energy","Healthcare","Industrials","Consumer","Real Estate","Utilities","Materials","Telecom"]

def build_portfolio(n=80):
    rows = []
    for i in range(n):
        r = random.random()
        if r < 0.5:
            rows.append(dict(
                position_id=f"L{i:03d}",
                asset_class="loan", sector=random.choice(SECTORS),
                rating=random.choice(["BBB","BB","B","CCC"]),
                notional=round(random.uniform(5e6, 120e6), 0),
                duration=round(random.uniform(1.0, 5.0), 2),
                pd=round(random.uniform(0.005, 0.08), 4),
                lgd=round(random.uniform(0.35, 0.60), 3),
                beta=0.2, delta=0.0, vega=0.0,
            ))
        elif r < 0.85:
            rows.append(dict(
                position_id=f"B{i:03d}",
                asset_class="bond", sector=random.choice(SECTORS),
                rating=random.choice(RATINGS),
                notional=round(random.uniform(5e6, 200e6), 0),
                duration=round(random.uniform(2.0, 9.0), 2),
                convexity=round(random.uniform(0.2, 1.5), 3),
                pd=0.0, lgd=0.0,
                beta=0.0, delta=0.0, vega=0.0,
            ))
        else:
            rows.append(dict(
                position_id=f"D{i:03d}",
                asset_class="derivative", sector=random.choice(SECTORS),
                rating="NR",
                notional=round(random.uniform(10e6, 300e6), 0),
                duration=0.0, pd=0.0, lgd=0.0,
                beta=0.0,
                delta=round(random.uniform(-1.2, 1.2), 2),
                vega=round(random.uniform(-0.5e6, 0.5e6), 0),
            ))
    df = pd.DataFrame(rows)
    for c in ["convexity","pd","lgd","beta","delta","vega"]:
        if c not in df.columns:
            df[c] = 0.0
    return df.fillna(0.0)

EVAL = [
    ("Fed signals two rate cuts", "Macroeconomic"),
    ("JPMorgan beats earnings", "Earnings"),
    ("Oil spikes as Middle East tensions escalate", "Geopolitical"),
    ("Apple unveils new headset", "Product Launch"),
    ("Credit Suisse bonds downgraded to junk", "Credit Event"),
    ("Microsoft to acquire AI startup", "Merger/Acquisition"),
    ("Tesla recalls 200k vehicles", "Operational"),
    ("Nvidia unveils Blackwell Ultra chips", "Product Launch"),
    ("US CPI hotter than expected", "Macroeconomic"),
    ("Boeing faces FAA probe", "Regulatory"),
    ("Amazon to spin off AWS logistics", "Merger/Acquisition"),
    ("China stimulus lifts markets", "Macroeconomic"),
    ("SVB Financial files for bankruptcy", "Credit Event"),
    ("Google antitrust ruling", "Regulatory"),
    ("J&J wins FDA approval", "Product Launch"),
    ("Walmart raises guidance", "Earnings"),
    ("Visa faces new interchange rules", "Regulatory"),
    ("Exxon announces buyback", "Earnings"),
    ("P&G warns of margin pressure", "Earnings"),
    ("UnitedHealth CEO steps down", "Operational"),
    ("Home Depot cuts outlook", "Earnings"),
    ("BofA raises dividend", "Earnings"),
    ("Meta fined by EU", "Regulatory"),
    ("Russia-Ukraine ceasefire talks stall", "Geopolitical"),
    ("Taiwan strait tensions rise", "Geopolitical"),
    ("OPEC+ cuts production", "Macroeconomic"),
    ("US jobs report blowout", "Macroeconomic"),
    ("CRE outlook downgraded", "Credit Event"),
    ("Ford and GM warn of EV slowdown", "Earnings"),
    ("Biotech IPO window reopens", "Macroeconomic"),
]

def main():
    build_news_df().to_csv(DATA / "news_sample.csv", index=False)
    build_tweets_df().to_csv(DATA / "tweets_sample.csv", index=False)
    print("[data] news_sample.csv, tweets_sample.csv written")

    from src.config import CONFIG
    tickers = CONFIG["module_a"]["tickers"]
    prices = build_prices_df(tickers)
    prices.to_csv(DATA / "prices.csv")
    print(f"[data] prices.csv written: {prices.shape}")

    port = build_portfolio(80)
    port.to_csv(DATA / "portfolio_positions.csv", index=False)
    print(f"[data] portfolio_positions.csv written: {port.shape}")

    pd.DataFrame(EVAL, columns=["text","label"]).to_csv(DATA / "labeled_eval.csv", index=False)
    print(f"[data] labeled_eval.csv written: {len(EVAL)} items")

    import yaml
    priors = {
        "Geopolitical": 0.8, "Credit Event": 0.9, "Macroeconomic": 0.7,
        "Merger/Acquisition": 0.6, "Regulatory": 0.55, "Earnings": 0.4,
        "Product Launch": 0.3, "Operational": 0.5, "Other": 0.3,
    }
    with open(DATA / "event_priors.yaml", "w") as f:
        yaml.safe_dump(priors, f)
    print("[data] event_priors.yaml written")


if __name__ == "__main__":
    main()
