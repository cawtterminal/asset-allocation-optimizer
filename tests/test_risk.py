import numpy as np
import pandas as pd
import pytest

from asset_allocation_optimizer import risk


def test_annualized_return_compounds_daily_returns(constant_returns):
    assert risk.annualized_return(constant_returns) == pytest.approx(1.001**252 - 1)


def test_annualized_return_is_minus_one_after_total_loss():
    assert risk.annualized_return(pd.Series([0.1, -1.0, 0.2])) == -1.0


def test_volatility_is_zero_without_dispersion(constant_returns):
    assert risk.annualized_volatility(constant_returns) == pytest.approx(0.0)


def test_volatility_scales_with_square_root_of_time():
    returns = pd.Series([0.01, -0.01] * 50)
    expected = returns.std(ddof=1) * np.sqrt(252)
    assert risk.annualized_volatility(returns) == pytest.approx(expected)


def test_sharpe_is_neutral_for_flat_returns(constant_returns):
    assert risk.sharpe_ratio(constant_returns) == 0.0


def test_sharpe_matches_manual_formula():
    returns = pd.Series([0.02, -0.01, 0.015, 0.0, -0.005])
    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    assert risk.sharpe_ratio(returns) == pytest.approx(expected)


def test_sortino_ignores_upside_volatility():
    calm = pd.Series([0.01, -0.005] * 20)
    volatile_upside = pd.Series([0.05, -0.005] * 20)
    assert risk.sortino_ratio(volatile_upside) > risk.sortino_ratio(calm)


def test_max_drawdown_finds_deepest_decline():
    returns = pd.Series([0.10, -0.50, 0.20, 0.30])
    assert risk.max_drawdown(returns) == pytest.approx(0.5)


def test_drawdown_counts_losses_from_the_initial_capital():
    drawdown = risk.drawdown_series(pd.Series([-0.10, 0.05]))
    assert drawdown.iloc[0] == pytest.approx(-0.10)
    assert drawdown.iloc[1] == pytest.approx(0.9 * 1.05 - 1)


def test_historical_var_and_cvar_on_known_distribution():
    returns = pd.Series(np.linspace(-0.10, 0.09, 20))
    var = risk.value_at_risk(returns, 0.90)
    cvar = risk.conditional_value_at_risk(returns, 0.90)
    assert var == pytest.approx(-np.quantile(returns, 0.10))
    assert cvar >= var


def test_parametric_var_uses_normal_quantile():
    returns = pd.Series(np.random.default_rng(1).normal(0, 0.01, 5000))
    assert risk.value_at_risk(returns, 0.95, "parametric") == pytest.approx(0.01645, abs=1e-3)


def test_var_rejects_unknown_method_and_bad_confidence():
    returns = pd.Series([0.01, -0.02, 0.03])
    with pytest.raises(ValueError):
        risk.value_at_risk(returns, method="magic")
    with pytest.raises(ValueError):
        risk.value_at_risk(returns, confidence=1.5)


def test_beta_of_leveraged_series_is_the_leverage():
    market = pd.Series(np.random.default_rng(2).normal(0, 0.01, 500))
    assert risk.beta(market * 2, market) == pytest.approx(2.0)


def test_alpha_is_zero_when_tracking_the_benchmark():
    market = pd.Series(np.random.default_rng(3).normal(0.0005, 0.01, 500))
    assert risk.jensen_alpha(market, market) == pytest.approx(0.0, abs=1e-12)


def test_information_ratio_is_zero_without_active_risk():
    market = pd.Series(np.random.default_rng(4).normal(0, 0.01, 100))
    assert risk.information_ratio(market, market) == 0.0


def test_risk_report_includes_relative_metrics_only_with_benchmark(returns):
    standalone = risk.risk_report(returns["US_EQUITY"])
    relative = risk.risk_report(returns["US_EQUITY"], returns["MARKET"])
    assert standalone.beta is None
    assert relative.beta == pytest.approx(risk.beta(returns["US_EQUITY"], returns["MARKET"]))
    assert set(relative.to_dict()) >= {"sharpe", "max_drawdown", "alpha"}
