from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

PERIODS_PER_YEAR = 252


def annualized_return(returns: pd.Series, periods: int = PERIODS_PER_YEAR) -> float:
    if returns.empty:
        return 0.0
    growth = float((1 + returns).prod())
    if growth <= 0:
        return -1.0
    return growth ** (periods / len(returns)) - 1


def annualized_volatility(returns: pd.Series, periods: int = PERIODS_PER_YEAR) -> float:
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(periods))


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0, periods: int = PERIODS_PER_YEAR) -> float:
    excess = returns - _per_period(risk_free, periods)
    deviation = excess.std(ddof=1)
    if len(returns) < 2 or np.isnan(deviation) or deviation < 1e-12:
        return 0.0
    return float(excess.mean() / deviation * np.sqrt(periods))


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0, periods: int = PERIODS_PER_YEAR) -> float:
    excess = returns - _per_period(risk_free, periods)
    downside = np.sqrt((np.minimum(excess, 0) ** 2).mean())
    if np.isnan(downside) or downside < 1e-12:
        return 0.0
    return float(excess.mean() / downside * np.sqrt(periods))


def drawdown_series(returns: pd.Series) -> pd.Series:
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax().clip(lower=1.0)
    return wealth / peak - 1


def max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    return float(-drawdown_series(returns).min())


def calmar_ratio(returns: pd.Series, periods: int = PERIODS_PER_YEAR) -> float:
    drawdown = max_drawdown(returns)
    return annualized_return(returns, periods) / drawdown if drawdown > 0 else 0.0


def value_at_risk(returns: pd.Series, confidence: float = 0.95, method: str = "historical") -> float:
    _check_confidence(confidence)
    if method == "historical":
        return float(-np.quantile(returns, 1 - confidence))
    if method == "parametric":
        z = norm.ppf(1 - confidence)
        return float(-(returns.mean() + z * returns.std(ddof=1)))
    raise ValueError(f"Unknown VaR method '{method}'. Use 'historical' or 'parametric'")


def conditional_value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    _check_confidence(confidence)
    threshold = np.quantile(returns, 1 - confidence)
    tail = returns[returns <= threshold]
    return float(-tail.mean()) if not tail.empty else 0.0


def beta(returns: pd.Series, benchmark: pd.Series) -> float:
    aligned = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return 0.0
    variance = aligned.iloc[:, 1].var(ddof=1)
    if variance == 0:
        return 0.0
    return float(aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / variance)


def jensen_alpha(
    returns: pd.Series, benchmark: pd.Series, risk_free: float = 0.0, periods: int = PERIODS_PER_YEAR
) -> float:
    aligned = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    rf = _per_period(risk_free, periods)
    portfolio, market = aligned.iloc[:, 0], aligned.iloc[:, 1]
    excess = (portfolio.mean() - rf) - beta(portfolio, market) * (market.mean() - rf)
    return float(excess * periods)


def tracking_error(returns: pd.Series, benchmark: pd.Series, periods: int = PERIODS_PER_YEAR) -> float:
    active = (returns - benchmark).dropna()
    return annualized_volatility(active, periods)


def information_ratio(returns: pd.Series, benchmark: pd.Series, periods: int = PERIODS_PER_YEAR) -> float:
    active = (returns - benchmark).dropna()
    error = tracking_error(returns, benchmark, periods)
    return float(active.mean() * periods / error) if error > 0 else 0.0


@dataclass(frozen=True)
class RiskReport:
    annual_return: float
    annual_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    var_95: float
    cvar_95: float
    best_day: float
    worst_day: float
    beta: float | None = None
    alpha: float | None = None
    tracking_error: float | None = None
    information_ratio: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


def risk_report(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    risk_free: float = 0.0,
    periods: int = PERIODS_PER_YEAR,
) -> RiskReport:
    relative = {}
    if benchmark is not None:
        relative = {
            "beta": beta(returns, benchmark),
            "alpha": jensen_alpha(returns, benchmark, risk_free, periods),
            "tracking_error": tracking_error(returns, benchmark, periods),
            "information_ratio": information_ratio(returns, benchmark, periods),
        }
    return RiskReport(
        annual_return=annualized_return(returns, periods),
        annual_volatility=annualized_volatility(returns, periods),
        sharpe=sharpe_ratio(returns, risk_free, periods),
        sortino=sortino_ratio(returns, risk_free, periods),
        max_drawdown=max_drawdown(returns),
        calmar=calmar_ratio(returns, periods),
        var_95=value_at_risk(returns, 0.95),
        cvar_95=conditional_value_at_risk(returns, 0.95),
        best_day=float(returns.max()),
        worst_day=float(returns.min()),
        **relative,
    )


def _per_period(annual_rate: float, periods: int) -> float:
    return (1 + annual_rate) ** (1 / periods) - 1


def _check_confidence(confidence: float) -> None:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
