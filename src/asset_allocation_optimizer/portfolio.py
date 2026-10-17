from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from asset_allocation_optimizer.risk import PERIODS_PER_YEAR


def normalize_weights(weights: Mapping[str, float] | pd.Series) -> pd.Series:
    series = pd.Series(weights, dtype=float)
    if series.empty:
        raise ValueError("At least one weight is required")
    if (series < 0).any():
        raise ValueError("Weights must be non-negative")
    total = series.sum()
    if total <= 0:
        raise ValueError("Weights must sum to a positive value")
    return series / total


def equal_weights(tickers: list[str]) -> pd.Series:
    return pd.Series(1.0 / len(tickers), index=tickers)


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    missing = sorted(set(weights.index) - set(returns.columns))
    if missing:
        raise ValueError(f"Missing returns for: {', '.join(missing)}")
    return returns[weights.index].dot(weights).rename("Portfolio")


def expected_returns(returns: pd.DataFrame, periods: int = PERIODS_PER_YEAR) -> pd.Series:
    return returns.mean() * periods


def covariance_matrix(returns: pd.DataFrame, periods: int = PERIODS_PER_YEAR) -> pd.DataFrame:
    return returns.cov() * periods


def portfolio_performance(
    weights: np.ndarray, mean_returns: np.ndarray, covariance: np.ndarray, risk_free: float = 0.0
) -> tuple[float, float, float]:
    expected = float(weights @ mean_returns)
    volatility = float(np.sqrt(max(weights @ covariance @ weights, 0.0)))
    sharpe = (expected - risk_free) / volatility if volatility > 0 else 0.0
    return expected, volatility, sharpe


def risk_contributions(weights: pd.Series, covariance: pd.DataFrame) -> pd.Series:
    w = weights.to_numpy()
    sigma = covariance.loc[weights.index, weights.index].to_numpy()
    variance = w @ sigma @ w
    if variance == 0:
        return pd.Series(0.0, index=weights.index)
    return pd.Series(w * (sigma @ w) / variance, index=weights.index)
