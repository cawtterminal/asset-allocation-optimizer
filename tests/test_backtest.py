import pandas as pd
import pytest

from asset_allocation_optimizer.backtest import RebalanceFrequency, compare_frequencies, rebalance_dates, run_backtest


@pytest.fixture
def two_assets() -> pd.DataFrame:
    index = pd.bdate_range("2024-01-01", "2024-12-31")
    rising = pd.Series([100 * 1.002**i for i in range(len(index))], index=index)
    flat = pd.Series(100.0, index=index)
    return pd.DataFrame({"UP": rising, "FLAT": flat})


def test_buy_and_hold_without_costs_matches_price_growth(two_assets):
    weights = pd.Series({"UP": 0.5, "FLAT": 0.5})
    result = run_backtest(two_assets, weights, RebalanceFrequency.NEVER, transaction_cost=0)
    growth = two_assets.iloc[-1] / two_assets.iloc[0]
    assert result.equity.iloc[-1] == pytest.approx(10_000 * (0.5 * growth["UP"] + 0.5 * growth["FLAT"]))
    assert result.rebalance_dates == []


def test_weights_drift_without_rebalancing(two_assets):
    result = run_backtest(two_assets, pd.Series({"UP": 0.5, "FLAT": 0.5}), RebalanceFrequency.NEVER, 10_000, 0)
    assert result.weights["UP"].iloc[-1] > 0.6


def test_rebalancing_restores_target_weights(two_assets):
    result = run_backtest(two_assets, pd.Series({"UP": 0.5, "FLAT": 0.5}), RebalanceFrequency.MONTHLY, 10_000, 0)
    last_rebalance = result.rebalance_dates[-1]
    assert result.weights.loc[last_rebalance, "UP"] == pytest.approx(0.5)


def test_rebalance_dates_are_last_trading_day_of_each_period(two_assets):
    monthly = rebalance_dates(two_assets.index, RebalanceFrequency.MONTHLY)
    quarterly = rebalance_dates(two_assets.index, RebalanceFrequency.QUARTERLY)
    assert len(monthly) == 11
    assert len(quarterly) == 3
    assert monthly[0] == pd.Timestamp("2024-01-31")
    assert rebalance_dates(two_assets.index, RebalanceFrequency.NEVER) == []


def test_transaction_costs_reduce_final_value(two_assets):
    weights = pd.Series({"UP": 0.5, "FLAT": 0.5})
    free = run_backtest(two_assets, weights, RebalanceFrequency.MONTHLY, transaction_cost=0)
    costly = run_backtest(two_assets, weights, RebalanceFrequency.MONTHLY, transaction_cost=0.01)
    assert costly.equity.iloc[-1] < free.equity.iloc[-1]
    assert costly.total_costs > 100


def test_weights_are_normalized(two_assets):
    result = run_backtest(two_assets, pd.Series({"UP": 2, "FLAT": 2}), RebalanceFrequency.NEVER, 10_000, 0)
    assert result.weights.iloc[0].tolist() == pytest.approx([0.5, 0.5])


def test_compare_frequencies_runs_every_policy(prices):
    assets = prices[["US_EQUITY", "BONDS"]]
    results = compare_frequencies(assets, pd.Series({"US_EQUITY": 0.6, "BONDS": 0.4}))
    assert set(results) == set(RebalanceFrequency)
    assert all(result.equity.notna().all() for result in results.values())


def test_invalid_inputs_are_rejected(two_assets):
    weights = pd.Series({"UP": 1.0})
    with pytest.raises(ValueError):
        run_backtest(two_assets, weights, initial_capital=0)
    with pytest.raises(ValueError):
        run_backtest(two_assets, weights, transaction_cost=0.5)
    with pytest.raises(ValueError):
        run_backtest(two_assets, pd.Series({"UP": -1.0, "FLAT": 2.0}))
