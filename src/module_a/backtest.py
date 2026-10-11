"""Backtest of Module A vs equal-weight benchmark."""
import numpy as np
import pandas as pd


def equal_weight_benchmark(prices: pd.DataFrame) -> pd.Series:
    rets = prices.pct_change().dropna()
    return (1 + rets.mean(axis=1)).cumprod()


def sentiment_backtest(prices: pd.DataFrame,
                       weights_history: pd.DataFrame):
    rets = prices.pct_change().fillna(0)
    common = rets.index.intersection(weights_history.index)
    if len(common) == 0:
        return pd.Series([1.0]), pd.Series([0.0])
    wr = weights_history.loc[common].shift(1).fillna(0)
    port_ret = (wr * rets.loc[common]).sum(axis=1)
    return (1 + port_ret).cumprod(), port_ret


def perf_stats(equity: pd.Series) -> dict:
    rets = equity.pct_change().dropna()
    if len(rets) < 2:
        return {"cum_return": 0.0, "ann_return": 0.0, "ann_vol": 0.0,
                "sharpe": 0.0, "max_drawdown": 0.0}
    ann = 252
    mean = float(rets.mean() * ann)
    vol = float(rets.std() * np.sqrt(ann))
    sharpe = mean / vol if vol else 0.0
    dd = float((equity / equity.cummax() - 1).min())
    return {
        "cum_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "ann_return": mean, "ann_vol": vol, "sharpe": sharpe,
        "max_drawdown": dd,
    }
