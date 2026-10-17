from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd


class DataError(ValueError):
    pass


class PriceProvider(Protocol):
    def fetch(self, tickers: list[str], start: date, end: date) -> pd.DataFrame: ...


class YahooFinanceProvider:
    def fetch(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        import yfinance as yf

        raw = yf.download(
            tickers=tickers,
            start=start.isoformat(),
            end=end.isoformat(),
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if raw is None or raw.empty:
            raise DataError("Yahoo Finance returned no data for " + ", ".join(tickers))

        close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
        if isinstance(close, pd.Series):
            close = close.to_frame(name=tickers[0])
        return clean_prices(close[[t for t in tickers if t in close.columns]])


class CsvProvider:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def fetch(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        prices = load_csv(self.path)
        missing = sorted(set(tickers) - set(prices.columns))
        if missing:
            raise DataError(f"Tickers not found in {self.path.name}: {', '.join(missing)}")
        window = prices.loc[pd.Timestamp(start) : pd.Timestamp(end), tickers]
        return clean_prices(window)


@dataclass(frozen=True)
class AssetProfile:
    ticker: str
    annual_return: float
    annual_volatility: float
    market_beta: float


class SyntheticProvider:
    def __init__(self, profiles: list[AssetProfile], seed: int = 38, start_price: float = 100.0) -> None:
        self.profiles = {profile.ticker: profile for profile in profiles}
        self.seed = seed
        self.start_price = start_price

    def fetch(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        unknown = sorted(set(tickers) - set(self.profiles))
        if unknown:
            raise DataError(f"No synthetic profile for: {', '.join(unknown)}")

        dates = pd.bdate_range(start, end)
        if len(dates) < 2:
            raise DataError("Date range must contain at least two business days")

        rng = np.random.default_rng(self.seed)
        periods = 252
        market = rng.normal(0.0, 1.0, len(dates))

        series = {}
        for ticker in tickers:
            profile = self.profiles[ticker]
            daily_vol = profile.annual_volatility / np.sqrt(periods)
            systematic_share = min(abs(profile.market_beta) * 0.16 / profile.annual_volatility, 0.95)
            idiosyncratic = rng.normal(0.0, 1.0, len(dates))
            shock = np.sign(profile.market_beta) * systematic_share * market
            shock += np.sqrt(1 - systematic_share**2) * idiosyncratic
            drift = np.log1p(profile.annual_return) / periods - daily_vol**2 / 2
            log_returns = drift + daily_vol * shock
            log_returns[0] = 0.0
            series[ticker] = self.start_price * np.exp(np.cumsum(log_returns))

        return pd.DataFrame(series, index=dates).round(4)


DEFAULT_SYNTHETIC_UNIVERSE = [
    AssetProfile("US_EQUITY", 0.10, 0.17, 1.0),
    AssetProfile("INTL_EQUITY", 0.07, 0.18, 0.9),
    AssetProfile("EM_EQUITY", 0.08, 0.24, 1.1),
    AssetProfile("TECH", 0.15, 0.28, 1.3),
    AssetProfile("BONDS", 0.035, 0.06, -0.1),
    AssetProfile("GOLD", 0.06, 0.15, 0.05),
    AssetProfile("REAL_ESTATE", 0.07, 0.20, 0.8),
    AssetProfile("MARKET", 0.09, 0.16, 1.0),
]


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DataError(f"File not found: {path}")
    frame = pd.read_csv(path)
    date_column = next((c for c in frame.columns if c.strip().lower() == "date"), None)
    if date_column is None:
        raise DataError(f"{path.name} must have a 'Date' column")
    frame[date_column] = pd.to_datetime(frame[date_column])
    frame = frame.set_index(date_column).sort_index()
    frame.index.name = "Date"
    return clean_prices(frame.apply(pd.to_numeric, errors="coerce"))


def save_csv(prices: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.rename_axis("Date").to_csv(path, float_format="%.4f")


def clean_prices(prices: pd.DataFrame, max_gap: int = 5) -> pd.DataFrame:
    if prices.empty:
        raise DataError("Price table is empty")
    cleaned = prices.sort_index()
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")]
    cleaned = cleaned.dropna(axis=1, how="all")
    cleaned = cleaned.ffill(limit=max_gap).dropna(how="any")
    if (cleaned <= 0).any().any():
        raise DataError("Prices must be strictly positive")
    if len(cleaned) < 2:
        raise DataError("Not enough overlapping history after cleaning")
    return cleaned


def to_returns(prices: pd.DataFrame, log: bool = False) -> pd.DataFrame:
    returns = np.log(prices / prices.shift(1)) if log else prices.pct_change(fill_method=None)
    return returns.dropna(how="all")
