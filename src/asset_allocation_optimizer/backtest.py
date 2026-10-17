from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from asset_allocation_optimizer.portfolio import normalize_weights


class RebalanceFrequency(StrEnum):
    NEVER = "never"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"

    @property
    def period_alias(self) -> str | None:
        return {"monthly": "M", "quarterly": "Q", "yearly": "Y"}.get(self.value)


@dataclass(frozen=True)
class BacktestResult:
    equity: pd.Series
    weights: pd.DataFrame
    rebalance_dates: list[pd.Timestamp]
    total_costs: float
    turnover: float

    @property
    def returns(self) -> pd.Series:
        return self.equity.pct_change().dropna()

    @property
    def total_return(self) -> float:
        return float(self.equity.iloc[-1] / self.equity.iloc[0] - 1)


def rebalance_dates(index: pd.DatetimeIndex, frequency: RebalanceFrequency) -> list[pd.Timestamp]:
    alias = frequency.period_alias
    if alias is None or len(index) < 2:
        return []
    periods = index.to_period(alias)
    is_period_end = periods[:-1] != periods[1:]
    return list(index[:-1][is_period_end])


def run_backtest(
    prices: pd.DataFrame,
    target_weights: pd.Series,
    frequency: RebalanceFrequency = RebalanceFrequency.QUARTERLY,
    initial_capital: float = 10_000.0,
    transaction_cost: float = 0.001,
) -> BacktestResult:
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    if not 0 <= transaction_cost < 0.1:
        raise ValueError("transaction_cost must be between 0 and 0.1")

    target = normalize_weights(target_weights)
    prices = prices[target.index].dropna()
    if len(prices) < 2:
        raise ValueError("At least two price observations are required")

    growth = (prices / prices.shift(1)).iloc[1:].to_numpy()
    rebalance_on = set(rebalance_dates(prices.index, frequency))
    target_vector = target.to_numpy()

    holdings = initial_capital * (1 - transaction_cost) * target_vector
    total_costs = initial_capital * transaction_cost
    total_turnover = 1.0

    equity = [holdings.sum()]
    allocation = [holdings / holdings.sum()]
    executed = []

    for step, day in enumerate(prices.index[1:]):
        holdings = holdings * growth[step]
        if day in rebalance_on:
            value = holdings.sum()
            trades = np.abs(target_vector * value - holdings).sum()
            cost = trades * transaction_cost
            holdings = target_vector * (value - cost)
            total_costs += cost
            total_turnover += trades / value / 2
            executed.append(day)
        equity.append(holdings.sum())
        allocation.append(holdings / holdings.sum())

    return BacktestResult(
        equity=pd.Series(equity, index=prices.index, name="Equity"),
        weights=pd.DataFrame(allocation, index=prices.index, columns=target.index),
        rebalance_dates=executed,
        total_costs=float(total_costs),
        turnover=float(total_turnover),
    )


def compare_frequencies(
    prices: pd.DataFrame,
    target_weights: pd.Series,
    initial_capital: float = 10_000.0,
    transaction_cost: float = 0.001,
) -> dict[RebalanceFrequency, BacktestResult]:
    return {
        frequency: run_backtest(prices, target_weights, frequency, initial_capital, transaction_cost)
        for frequency in RebalanceFrequency
    }
