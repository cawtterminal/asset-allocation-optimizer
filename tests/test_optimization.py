import numpy as np
import pandas as pd
import pytest

from asset_allocation_optimizer.optimization import (
    OptimizationConstraints,
    efficient_frontier,
    max_sharpe,
    min_variance,
    random_portfolios,
    risk_parity,
)
from asset_allocation_optimizer.portfolio import (
    covariance_matrix,
    equal_weights,
    portfolio_performance,
    risk_contributions,
)

OPTIMIZERS = [min_variance, max_sharpe, risk_parity]


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_weights_are_long_only_and_fully_invested(asset_returns, optimizer):
    weights = optimizer(asset_returns, 0.02).weights
    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0).all()
    assert list(weights.index) == list(asset_returns.columns)


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_max_weight_constraint_is_respected(asset_returns, optimizer):
    weights = optimizer(asset_returns, 0.02, OptimizationConstraints(max_weight=0.3)).weights
    assert weights.max() <= 0.3 + 1e-6


def test_min_variance_beats_every_random_portfolio(asset_returns):
    optimal = min_variance(asset_returns)
    cloud = random_portfolios(asset_returns, 2000)
    assert optimal.volatility <= cloud["volatility"].min() + 1e-9


def test_max_sharpe_beats_every_random_portfolio(asset_returns):
    optimal = max_sharpe(asset_returns, 0.02)
    cloud = random_portfolios(asset_returns, 2000, risk_free=0.02)
    assert optimal.sharpe >= cloud["sharpe"].max() - 1e-6


def test_min_variance_matches_closed_form_for_uncorrelated_assets():
    rng = np.random.default_rng(5)
    returns = pd.DataFrame({"A": rng.normal(0, 0.01, 5000), "B": rng.normal(0, 0.02, 5000)})
    inverse = np.linalg.inv(returns.cov().to_numpy())
    expected = inverse.sum(axis=1) / inverse.sum()
    weights = min_variance(returns).weights.to_numpy()
    assert weights == pytest.approx(expected, abs=1e-4)
    assert weights[0] == pytest.approx(0.8, abs=0.01)


def test_risk_parity_equalizes_risk_contributions(asset_returns):
    weights = risk_parity(asset_returns).weights
    contributions = risk_contributions(weights, covariance_matrix(asset_returns))
    assert contributions.to_numpy() == pytest.approx(np.full(len(weights), 1 / len(weights)), abs=1e-4)


def test_efficient_frontier_is_increasing_in_risk_and_return(asset_returns):
    frontier = efficient_frontier(asset_returns, points=15)
    assert len(frontier) >= 10
    assert frontier["return"].is_monotonic_increasing
    assert frontier["volatility"].diff().dropna().min() >= -1e-6


def test_random_portfolios_are_reproducible(asset_returns):
    first = random_portfolios(asset_returns, 100, seed=1)
    second = random_portfolios(asset_returns, 100, seed=1)
    pd.testing.assert_frame_equal(first, second)
    assert first[asset_returns.columns].sum(axis=1).to_numpy() == pytest.approx(np.ones(100))


def test_portfolio_performance_formula():
    mean = np.array([0.10, 0.05])
    covariance = np.array([[0.04, 0.0], [0.0, 0.01]])
    expected, volatility, sharpe = portfolio_performance(np.array([0.5, 0.5]), mean, covariance, 0.01)
    assert expected == pytest.approx(0.075)
    assert volatility == pytest.approx(np.sqrt(0.0125))
    assert sharpe == pytest.approx((0.075 - 0.01) / np.sqrt(0.0125))


def test_equal_weights_split_evenly():
    assert equal_weights(["A", "B", "C", "D"]).tolist() == [0.25] * 4


def test_infeasible_constraints_are_rejected(asset_returns):
    with pytest.raises(ValueError):
        min_variance(asset_returns, constraints=OptimizationConstraints(max_weight=0.2))
    with pytest.raises(ValueError):
        OptimizationConstraints(min_weight=0.5, max_weight=0.4)


def test_optimization_requires_two_assets(asset_returns):
    with pytest.raises(ValueError):
        max_sharpe(asset_returns[["BONDS"]])
