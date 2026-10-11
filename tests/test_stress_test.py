from src.module_b.stress_test import load_portfolio, portfolio_before_after


def test_stress_geopolitical():
    port = load_portfolio()
    res = portfolio_before_after(port, "Geopolitical")
    assert res["notional_before"] > 0
    assert res["notional_after"] != res["notional_before"]
    assert "by_asset_class" in res
