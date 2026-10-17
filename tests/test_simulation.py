import numpy as np
import pandas as pd
import pytest

from asset_allocation_optimizer.simulation import monte_carlo


@pytest.fixture
def history() -> pd.Series:
    return pd.Series(np.random.default_rng(9).normal(0.0004, 0.01, 750))


@pytest.mark.parametrize("method", ["bootstrap", "gbm"])
def test_paths_start_at_initial_value(history, method):
    result = monte_carlo(history, horizon_days=50, simulations=200, initial_value=5_000, method=method)
    assert result.paths.shape == (200, 51)
    assert np.all(result.paths[:, 0] == 5_000)
    assert np.all(result.paths > 0)


def test_simulation_is_reproducible_with_seed(history):
    first = monte_carlo(history, simulations=100, seed=1)
    second = monte_carlo(history, simulations=100, seed=1)
    assert np.array_equal(first.paths, second.paths)


def test_percentiles_are_ordered(history):
    bands = monte_carlo(history, simulations=1000).percentiles()
    final = bands.iloc[-1]
    assert final["p5"] < final["p25"] < final["p50"] < final["p75"] < final["p95"]


def test_constant_positive_returns_never_lose(constant_returns):
    result = monte_carlo(constant_returns, horizon_days=10, simulations=50)
    assert result.probability_of_loss() == 0.0
    assert result.expected_final_value() == pytest.approx(10_000 * 1.001**10)
    assert result.probability_of_reaching(10_000) == 1.0


def test_invalid_arguments_are_rejected(history):
    with pytest.raises(ValueError):
        monte_carlo(history, horizon_days=0)
    with pytest.raises(ValueError):
        monte_carlo(history, method="quantum")
    with pytest.raises(ValueError):
        monte_carlo(pd.Series([0.01]))
