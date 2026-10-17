from datetime import date

import numpy as np
import pandas as pd
import pytest

from asset_allocation_optimizer import data
from asset_allocation_optimizer.data import (
    DEFAULT_SYNTHETIC_UNIVERSE,
    CsvProvider,
    DataError,
    SyntheticProvider,
    YahooFinanceProvider,
    clean_prices,
    load_csv,
    save_csv,
    to_returns,
)


def test_csv_round_trip(tmp_path, prices):
    path = tmp_path / "prices.csv"
    save_csv(prices, path)
    loaded = load_csv(path)
    assert list(loaded.columns) == list(prices.columns)
    assert loaded.iloc[-1].to_numpy() == pytest.approx(prices.iloc[-1].to_numpy())


def test_csv_provider_filters_tickers_and_dates(tmp_path, prices):
    path = tmp_path / "prices.csv"
    save_csv(prices, path)
    window = CsvProvider(path).fetch(["BONDS", "GOLD"], date(2021, 1, 1), date(2021, 12, 31))
    assert list(window.columns) == ["BONDS", "GOLD"]
    assert window.index.min() >= pd.Timestamp("2021-01-01")
    assert window.index.max() <= pd.Timestamp("2021-12-31")


def test_csv_provider_reports_missing_tickers(tmp_path, prices):
    path = tmp_path / "prices.csv"
    save_csv(prices, path)
    with pytest.raises(DataError, match="NOPE"):
        CsvProvider(path).fetch(["NOPE"], date(2021, 1, 1), date(2021, 12, 31))


def test_load_csv_requires_date_column(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("Day,A\n2024-01-01,10\n2024-01-02,11\n")
    with pytest.raises(DataError):
        load_csv(path)


def test_clean_prices_fills_short_gaps_and_sorts():
    index = pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"])
    raw = pd.DataFrame({"A": [12.0, 10.0, np.nan], "B": [5.0, 4.0, 4.5]}, index=index)
    cleaned = clean_prices(raw)
    assert cleaned.index.is_monotonic_increasing
    assert cleaned.loc["2024-01-02", "A"] == 10.0


def test_clean_prices_rejects_non_positive_values():
    frame = pd.DataFrame({"A": [1.0, 0.0]}, index=pd.bdate_range("2024-01-01", periods=2))
    with pytest.raises(DataError):
        clean_prices(frame)


def test_returns_drop_first_observation():
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0]}, index=pd.bdate_range("2024-01-01", periods=3))
    returns = to_returns(prices)
    assert returns["A"].tolist() == pytest.approx([0.10, -0.10])
    assert to_returns(prices, log=True)["A"].iloc[0] == pytest.approx(np.log(1.1))


def test_synthetic_provider_is_deterministic_and_business_days_only():
    provider = SyntheticProvider(DEFAULT_SYNTHETIC_UNIVERSE, seed=7)
    first = provider.fetch(["TECH", "BONDS"], date(2024, 1, 1), date(2024, 6, 30))
    second = provider.fetch(["TECH", "BONDS"], date(2024, 1, 1), date(2024, 6, 30))
    pd.testing.assert_frame_equal(first, second)
    assert (first.index.dayofweek < 5).all()
    assert first.iloc[0].tolist() == [100.0, 100.0]


def test_synthetic_bonds_are_less_volatile_than_tech(returns):
    assert returns["BONDS"].std() < returns["TECH"].std() / 2


def test_yahoo_provider_handles_multi_ticker_response(monkeypatch):
    index = pd.bdate_range("2024-01-01", periods=3)
    columns = pd.MultiIndex.from_product([["Close", "Volume"], ["AAA", "BBB"]])
    frame = pd.DataFrame(np.full((3, 4), 10.0), index=index, columns=columns)

    fake_module = type("FakeYF", (), {"download": staticmethod(lambda **kwargs: frame)})
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_module)

    prices = YahooFinanceProvider().fetch(["AAA", "BBB"], date(2024, 1, 1), date(2024, 1, 5))
    assert list(prices.columns) == ["AAA", "BBB"]
    assert len(prices) == 3


def test_yahoo_provider_raises_on_empty_download(monkeypatch):
    fake_module = type("FakeYF", (), {"download": staticmethod(lambda **kwargs: pd.DataFrame())})
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_module)
    with pytest.raises(DataError):
        YahooFinanceProvider().fetch(["AAA"], date(2024, 1, 1), date(2024, 1, 5))


def test_sample_dataset_ships_with_the_repository():
    sample = load_csv(data.Path(__file__).resolve().parents[1] / "data" / "sample_prices.csv")
    assert {"US_EQUITY", "BONDS", "MARKET"} <= set(sample.columns)
    assert len(sample) > 1000
