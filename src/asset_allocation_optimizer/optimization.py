from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from asset_allocation_optimizer.portfolio import covariance_matrix, expected_returns, portfolio_performance


class OptimizationError(RuntimeError):
    pass


@dataclass(frozen=True)
class OptimizationConstraints:
    min_weight: float = 0.0
    max_weight: float = 1.0

    def __post_init__(self) -> None:
        if not 0 <= self.min_weight <= self.max_weight <= 1:
            raise ValueError("Constraints must satisfy 0 <= min_weight <= max_weight <= 1")

    def validate_for(self, n_assets: int) -> None:
        if self.max_weight * n_assets < 1 - 1e-9:
            raise ValueError(f"max_weight {self.max_weight:.0%} is too low to invest 100% in {n_assets} assets")
        if self.min_weight * n_assets > 1 + 1e-9:
            raise ValueError(f"min_weight {self.min_weight:.0%} is too high for {n_assets} assets")

    def bounds(self, n_assets: int) -> list[tuple[float, float]]:
        return [(self.min_weight, self.max_weight)] * n_assets


@dataclass(frozen=True)
class OptimalPortfolio:
    name: str
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe: float


@dataclass(frozen=True)
class _Inputs:
    tickers: list[str]
    mean: np.ndarray
    covariance: np.ndarray
    risk_free: float
    constraints: OptimizationConstraints


def min_variance(
    returns: pd.DataFrame,
    risk_free: float = 0.0,
    constraints: OptimizationConstraints | None = None,
) -> OptimalPortfolio:
    inputs = _prepare(returns, risk_free, constraints)
    weights = _solve(inputs, lambda w: w @ inputs.covariance @ w)
    return _result("Minimum Variance", weights, inputs)


def max_sharpe(
    returns: pd.DataFrame,
    risk_free: float = 0.0,
    constraints: OptimizationConstraints | None = None,
) -> OptimalPortfolio:
    inputs = _prepare(returns, risk_free, constraints)

    def negative_sharpe(w: np.ndarray) -> float:
        return -portfolio_performance(w, inputs.mean, inputs.covariance, inputs.risk_free)[2]

    weights = _solve(inputs, negative_sharpe)
    return _result("Maximum Sharpe", weights, inputs)


def risk_parity(
    returns: pd.DataFrame,
    risk_free: float = 0.0,
    constraints: OptimizationConstraints | None = None,
) -> OptimalPortfolio:
    inputs = _prepare(returns, risk_free, constraints)
    n_assets = len(inputs.tickers)
    budget = 1 / n_assets

    def barrier(y: np.ndarray) -> float:
        return 0.5 * y @ inputs.covariance @ y - budget * np.sum(np.log(y))

    def gradient(y: np.ndarray) -> np.ndarray:
        return inputs.covariance @ y - budget / y

    start = np.full(n_assets, 1 / np.sqrt(np.trace(inputs.covariance)))
    solution = minimize(barrier, start, jac=gradient, method="L-BFGS-B", bounds=[(1e-10, None)] * n_assets)
    if not solution.success:
        raise OptimizationError(solution.message)
    weights = solution.x / solution.x.sum()

    limits = inputs.constraints
    if weights.max() > limits.max_weight + 1e-9 or weights.min() < limits.min_weight - 1e-9:
        caps = [{"type": "ineq", "fun": lambda y, i=i: limits.max_weight * np.sum(y) - y[i]} for i in range(n_assets)]
        floors = [{"type": "ineq", "fun": lambda y, i=i: y[i] - limits.min_weight * np.sum(y)} for i in range(n_assets)]
        scale = solution.x.sum()
        clipped = np.clip(weights, max(limits.min_weight, 1e-4), limits.max_weight)
        constrained = minimize(
            barrier,
            clipped / clipped.sum() * scale,
            jac=gradient,
            method="SLSQP",
            bounds=[(1e-10, None)] * n_assets,
            constraints=caps + floors,
            options={"maxiter": 1000, "ftol": 1e-14},
        )
        candidate = constrained.x / constrained.x.sum()
        feasible = candidate.max() <= limits.max_weight + 1e-6 and candidate.min() >= limits.min_weight - 1e-6
        if not feasible:
            raise OptimizationError(constrained.message)
        weights = np.clip(candidate, limits.min_weight, limits.max_weight)
        weights = weights / weights.sum()
    return _result("Risk Parity", weights, inputs)


def efficient_frontier(
    returns: pd.DataFrame,
    points: int = 40,
    risk_free: float = 0.0,
    constraints: OptimizationConstraints | None = None,
) -> pd.DataFrame:
    if points < 2:
        raise ValueError("points must be at least 2")
    inputs = _prepare(returns, risk_free, constraints)
    lowest = _result("", _solve(inputs, lambda w: w @ inputs.covariance @ w), inputs).expected_return
    highest = _result("", _solve(inputs, lambda w: -(w @ inputs.mean)), inputs).expected_return

    rows = []
    for target in np.linspace(lowest, highest, points):
        target_constraint = {"type": "eq", "fun": lambda w, t=target: w @ inputs.mean - t}
        try:
            weights = _solve(inputs, lambda w: w @ inputs.covariance @ w, [target_constraint])
        except OptimizationError:
            continue
        expected, volatility, sharpe = portfolio_performance(weights, inputs.mean, inputs.covariance, inputs.risk_free)
        rows.append(
            {
                "return": expected,
                "volatility": volatility,
                "sharpe": sharpe,
                **dict(zip(inputs.tickers, weights, strict=True)),
            }
        )
    return pd.DataFrame(rows)


def random_portfolios(
    returns: pd.DataFrame,
    count: int = 3000,
    risk_free: float = 0.0,
    seed: int = 7,
) -> pd.DataFrame:
    tickers = list(returns.columns)
    mean = expected_returns(returns).to_numpy()
    covariance = covariance_matrix(returns).to_numpy()
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(len(tickers)), size=count)

    expected = weights @ mean
    volatility = np.sqrt(np.einsum("ij,jk,ik->i", weights, covariance, weights))
    sharpe = np.divide(expected - risk_free, volatility, out=np.zeros(count), where=volatility > 0)

    frame = pd.DataFrame(weights, columns=tickers)
    frame.insert(0, "sharpe", sharpe)
    frame.insert(0, "volatility", volatility)
    frame.insert(0, "return", expected)
    return frame


def _prepare(returns: pd.DataFrame, risk_free: float, constraints: OptimizationConstraints | None) -> _Inputs:
    if returns.shape[1] < 2:
        raise ValueError("Optimization needs at least two assets")
    if len(returns) < returns.shape[1] + 2:
        raise ValueError("Not enough observations to estimate the covariance matrix")
    constraints = constraints or OptimizationConstraints()
    constraints.validate_for(returns.shape[1])
    return _Inputs(
        tickers=list(returns.columns),
        mean=expected_returns(returns).to_numpy(),
        covariance=covariance_matrix(returns).to_numpy(),
        risk_free=risk_free,
        constraints=constraints,
    )


def _solve(
    inputs: _Inputs,
    objective,
    extra_constraints: list[dict] | None = None,
    initial: np.ndarray | None = None,
) -> np.ndarray:
    n_assets = len(inputs.tickers)
    if initial is None:
        initial = np.clip(np.full(n_assets, 1 / n_assets), inputs.constraints.min_weight, inputs.constraints.max_weight)
    budget = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=inputs.constraints.bounds(n_assets),
        constraints=[budget, *(extra_constraints or [])],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    all_constraints = [budget, *(extra_constraints or [])]
    if not result.success and not _is_precision_limit(result, inputs, all_constraints):
        raise OptimizationError(result.message)
    weights = np.clip(result.x, 0, None)
    weights[weights < 1e-6] = 0
    return weights / weights.sum()


def _is_precision_limit(result, inputs: _Inputs, constraints: list[dict], tolerance: float = 1e-6) -> bool:
    if result.status != 8:
        return False
    within_bounds = np.all(result.x >= inputs.constraints.min_weight - tolerance) and np.all(
        result.x <= inputs.constraints.max_weight + tolerance
    )
    equalities_hold = all(abs(constraint["fun"](result.x)) < tolerance for constraint in constraints)
    return bool(within_bounds and equalities_hold)


def _result(name: str, weights: np.ndarray, inputs: _Inputs) -> OptimalPortfolio:
    expected, volatility, sharpe = portfolio_performance(weights, inputs.mean, inputs.covariance, inputs.risk_free)
    return OptimalPortfolio(
        name=name,
        weights=pd.Series(weights, index=inputs.tickers),
        expected_return=expected,
        volatility=volatility,
        sharpe=sharpe,
    )
