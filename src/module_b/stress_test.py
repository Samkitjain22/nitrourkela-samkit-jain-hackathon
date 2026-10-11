"""Event-driven stress test on a synthetic wholesale-banking portfolio (Module B)."""
import pandas as pd
from src.module_b.shocks import SHOCKS
from src.config import CONFIG


def load_portfolio() -> pd.DataFrame:
    return pd.read_csv(CONFIG["paths"]["portfolio_csv"])


def _bond_pnl(row, rate_bp, spread_bp):
    d = float(row.get("duration", 0.0) or 0.0)
    c = float(row.get("convexity", 0.0) or 0.0)
    dy = (rate_bp + spread_bp) / 10000.0
    pct = -d * dy + 0.5 * c * dy ** 2
    return float(row["notional"]) * pct


def _equity_pnl(row, eq_pct):
    return float(row["notional"]) * eq_pct * float(row.get("beta", 1.0) or 1.0)


def _loan_pnl(row, spread_bp, pd_mult=1.0):
    pd_ = float(row.get("pd", 0.02) or 0.02)
    lgd = float(row.get("lgd", 0.45) or 0.45)
    ead = float(row["notional"])
    base_el = pd_ * lgd * ead
    d = float(row.get("duration", 2.0) or 2.0)
    mtm = ead * (-d) * (spread_bp / 10000.0)
    return -(mtm + base_el * (pd_mult - 1))


def apply_shock(portfolio: pd.DataFrame, shock: dict, name: str) -> pd.DataFrame:
    pnl = []
    for _, r in portfolio.iterrows():
        ac = r["asset_class"]
        if ac == "equity":
            pnl.append(_equity_pnl(r, shock.get("equity_pct", 0.0)))
        elif ac == "bond":
            pnl.append(_bond_pnl(r, shock.get("rates_bp", 0),
                                 shock.get("credit_spread_bp", 0)))
        elif ac == "loan":
            pnl.append(_loan_pnl(r, shock.get("credit_spread_bp", 0),
                                 shock.get("default_prob_mult", 1.0)))
        elif ac == "derivative":
            delta = float(r.get("delta", 0.0) or 0.0)
            vega = float(r.get("vega", 0.0) or 0.0)
            dv = float(r["notional"]) * delta * shock.get("equity_pct", 0.0)
            pnl.append(dv + vega * shock.get("credit_spread_bp", 0) * 0.0001)
        else:
            pnl.append(0.0)
    df = portfolio.copy()
    df["pnl"] = pnl
    df["shock"] = name
    return df


def portfolio_before_after(portfolio: pd.DataFrame, event_type: str):
    if event_type not in SHOCKS:
        event_type = "Other"
    shocked = apply_shock(portfolio, SHOCKS[event_type], event_type)
    notional_before = float(portfolio["notional"].sum())
    total_pnl = float(shocked["pnl"].sum())
    return {
        "event": event_type,
        "shock": SHOCKS[event_type],
        "notional_before": notional_before,
        "total_pnl": total_pnl,
        "notional_after": notional_before + total_pnl,
        "pct_change": total_pnl / notional_before if notional_before else 0.0,
        "top_affected": shocked.reindex(
            shocked["pnl"].abs().sort_values(ascending=False).index
        ).head(10),
        "by_asset_class": shocked.groupby("asset_class")["pnl"].sum().reset_index(),
        "detail": shocked,
    }
