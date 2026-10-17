from datetime import date

import numpy as np
import pandas as pd
import pytest

from asset_allocation_optimizer.data import DEFAULT_SYNTHETIC_UNIVERSE, SyntheticProvider, to_returns

TICKERS = ["US_EQUITY", "TECH", "BONDS", "GOLD"]


@pytest.fixture(scope="session")
def prices() -> pd.DataFrame:
    provider = SyntheticProvider(DEFAULT_SYNTHETIC_UNIVERSE, seed=38)
    return provider.fetch([*TICKERS, "MARKET"], date(2020, 1, 1), date(2023, 12, 31))


@pytest.fixture(scope="session")
def returns(prices: pd.DataFrame) -> pd.DataFrame:
    return to_returns(prices)


@pytest.fixture(scope="session")
def asset_returns(returns: pd.DataFrame) -> pd.DataFrame:
    return returns[TICKERS]


@pytest.fixture
def constant_returns() -> pd.Series:
    index = pd.bdate_range("2024-01-01", periods=252)
    return pd.Series(np.full(252, 0.001), index=index)
