from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulationResult:
    paths: np.ndarray
    initial_value: float

    @property
    def final_values(self) -> np.ndarray:
        return self.paths[:, -1]

    def percentiles(self, levels: tuple[int, ...] = (5, 25, 50, 75, 95)) -> pd.DataFrame:
        bands = np.percentile(self.paths, levels, axis=0)
        return pd.DataFrame(bands.T, columns=[f"p{level}" for level in levels])

    def probability_of_loss(self) -> float:
        return float(np.mean(self.final_values < self.initial_value))

    def probability_of_reaching(self, target: float) -> float:
        return float(np.mean(self.final_values >= target))

    def expected_final_value(self) -> float:
        return float(np.mean(self.final_values))


def monte_carlo(
    returns: pd.Series,
    horizon_days: int = 252,
    simulations: int = 5000,
    initial_value: float = 10_000.0,
    method: str = "bootstrap",
    seed: int | None = 42,
) -> SimulationResult:
    if horizon_days <= 0 or simulations <= 0:
        raise ValueError("horizon_days and simulations must be positive")
    if initial_value <= 0:
        raise ValueError("initial_value must be positive")
    history = returns.dropna().to_numpy()
    if len(history) < 2:
        raise ValueError("At least two historical returns are required")

    rng = np.random.default_rng(seed)
    if method == "bootstrap":
        sampled = rng.choice(history, size=(simulations, horizon_days), replace=True)
        log_growth = np.log1p(sampled)
    elif method == "gbm":
        log_history = np.log1p(history)
        mu, sigma = log_history.mean(), log_history.std(ddof=1)
        log_growth = rng.normal(mu, sigma, size=(simulations, horizon_days))
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'bootstrap' or 'gbm'")

    cumulative = np.cumsum(log_growth, axis=1)
    paths = initial_value * np.exp(np.hstack([np.zeros((simulations, 1)), cumulative]))
    return SimulationResult(paths=paths, initial_value=initial_value)
