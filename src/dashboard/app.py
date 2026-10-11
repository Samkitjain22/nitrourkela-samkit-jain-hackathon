"""Streamlit dashboard. Run: streamlit run src/dashboard/app.py"""
import os
import sys

# Add project root to sys.path so "from src.xxx import" works when run via streamlit
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import CONFIG, ROOT
from src.engine.risk_engine import RiskEngine
from src.module_a.rebalancer import SentimentRebalancer
from src.module_a.backtest import equal_weight_benchmark, sentiment_backtest, perf_stats
from src.module_b.stress_test import load_portfolio, portfolio_before_after
from src.module_b.shocks import SHOCKS

st.set_page_config(page_title="AI/NLP Risk Engine", layout="wide")
st.title("AI/NLP Risk Engine — S&P Global & Crisil Hackathon 2026")


@st.cache_resource
def get_engine():
    return RiskEngine()


@st.cache_data
def load_signals():
    path = Path(CONFIG["paths"]["signals_csv"])
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


@st.cache_data
def load_prices():
    p = Path(CONFIG["paths"]["prices_csv"])
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    return df


@st.cache_data
def load_eval():
    return pd.read_csv(CONFIG["paths"]["eval_csv"])


tabs = st.tabs(["Live Signals", "Rebalancer (Module A)",
                "Stress Test (Module B)", "Model Evaluation", "Try a Headline"])

with tabs[0]:
    st.subheader("Live structured risk signals")
    df = load_signals()
    if df.empty:
        st.warning("No signals yet. Run `python -m src.run_pipeline`.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Signals", len(df))
        c2.metric("High-impact (>7)", int((df["impact"] > 7).sum()))
        c3.metric("Avg sentiment", f"{df['sentiment'].mean():.2f}")
        f1, f2 = st.columns(2)
        ev = f1.multiselect("Filter event", sorted(df["event"].unique()),
                            default=list(df["event"].unique()))
        src = f2.multiselect("Filter source", sorted(df["source"].unique()),
                             default=list(df["source"].unique()))
        view = df[df["event"].isin(ev) & df["source"].isin(src)]
        st.dataframe(view.sort_values("timestamp", ascending=False)[
            ["timestamp", "source", "text", "tickers", "sentiment", "event", "impact"]
        ].head(200), use_container_width=True)
        fig = px.histogram(view, x="impact", color="event",
                           nbins=20, title="Impact score distribution")
        st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.subheader("Sentiment-driven index rebalancer")
    df = load_signals()
    prices = load_prices()
    reb = SentimentRebalancer(CONFIG)
    if df.empty or prices.empty:
        st.warning("Run `python -m src.run_pipeline` and ensure prices.csv exists.")
    else:
        sent = reb.decayed_sentiment(
            df, half_life_hours=CONFIG["module_a"]["decay_half_life_hours"])
        weights = reb.compute_weights(sent)
        st.markdown("#### Current weights (tilted by impact-weighted sentiment)")
        wdf = weights.reset_index()
        wdf.columns = ["ticker", "weight"]
        wdf["sentiment"] = wdf["ticker"].map(sent).fillna(0.0)
        fig = px.bar(wdf.sort_values("weight", ascending=False),
                     x="ticker", y="weight", color="sentiment",
                     color_continuous_scale="RdYlGn", title="Portfolio weights")
        st.plotly_chart(fig, use_container_width=True)
        tickers = CONFIG["module_a"]["tickers"]
        p = prices[tickers].dropna(how="all").ffill().dropna()
        wh = pd.DataFrame([weights.reindex(tickers).fillna(0)] * len(p),
                          index=p.index, columns=tickers)
        eq_sent, _ = sentiment_backtest(p, wh)
        eq_bench = equal_weight_benchmark(p)
        st.markdown("#### Backtest vs equal-weight benchmark")
        chart = pd.DataFrame({"Sentiment-Tilted": eq_sent, "Equal-Weight": eq_bench})
        st.line_chart(chart)
        s1, s2 = perf_stats(eq_sent), perf_stats(eq_bench)
        st.table(pd.DataFrame({"Sentiment-Tilted": s1, "Equal-Weight": s2}))
        st.caption("Weights held constant here for apples-to-apples comparison.")

with tabs[2]:
    st.subheader("Event-driven stress test")
    try:
        port = load_portfolio()
    except FileNotFoundError:
        port = None
        st.warning("portfolio_positions.csv missing — run bootstrap_data.")
    if port is not None:
        df = load_signals()
        triggers = CONFIG["module_b"]["trigger_events"]
        auto = None
        if not df.empty:
            hot = df[(df["impact"] > CONFIG["module_b"]["impact_threshold"]) &
                     (df["event"].isin(triggers))]
            if not hot.empty:
                auto = hot.sort_values("impact", ascending=False).iloc[0]
        if auto is not None:
            st.info(f"Auto-trigger: **{auto['event']}** (impact {auto['impact']}) "
                    f"from `{auto['text'][:80]}...`")
        event_choice = st.selectbox(
            "Trigger event", options=list(SHOCKS.keys()),
            index=list(SHOCKS.keys()).index(auto["event"]) if auto is not None else 0,
        )
        res = portfolio_before_after(port, event_choice)
        c1, c2, c3 = st.columns(3)
        c1.metric("Notional before", f"${res['notional_before']/1e6:,.1f}M")
        c2.metric("P&L", f"${res['total_pnl']/1e6:,.1f}M",
                  delta=f"{res['pct_change']*100:.2f}%")
        c3.metric("Notional after", f"${res['notional_after']/1e6:,.1f}M")
        st.markdown("#### P&L by asset class")
        fig = px.bar(res["by_asset_class"], x="asset_class", y="pnl",
                     color="asset_class")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("#### Top 10 most affected positions")
        st.dataframe(res["top_affected"][
            ["position_id", "asset_class", "sector", "rating", "notional", "pnl"]
        ], use_container_width=True)
        st.markdown("#### Shock parameters applied")
        st.json(res["shock"])

with tabs[3]:
    st.subheader("Model evaluation on hand-labeled set")
    ev = load_eval()
    engine = get_engine()
    preds = [engine.evt.classify(t)["event"] for t in ev["text"]]
    labels = sorted(set(ev["label"]) | set(preds))
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    acc = accuracy_score(ev["label"], preds)
    macro_f1 = f1_score(ev["label"], preds, average="macro", labels=labels,
                        zero_division=0)
    st.metric("Event classification accuracy", f"{acc:.3f}")
    st.metric("Macro-F1", f"{macro_f1:.3f}")
    cm = confusion_matrix(ev["label"], preds, labels=labels)
    fig = px.imshow(cm, x=labels, y=labels, text_auto=True,
                    labels=dict(x="Predicted", y="True"),
                    title="Confusion matrix (event classifier)")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(pd.DataFrame({"text": ev["text"], "true": ev["label"], "pred": preds}))

with tabs[4]:
    st.subheader("Paste any headline → structured signal")
    text = st.text_area("Headline / tweet",
                        "Fed hikes rates by 50bp as inflation hits 5%")
    src = st.text_input("Source", "reuters")
    if st.button("Analyze"):
        engine = get_engine()
        sig = engine.analyze_one(src, text, datetime.now(timezone.utc))
        st.json(sig.model_dump())
