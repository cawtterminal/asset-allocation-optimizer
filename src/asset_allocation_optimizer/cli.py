from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from asset_allocation_optimizer.backtest import compare_frequencies
from asset_allocation_optimizer.data import (
    DEFAULT_SYNTHETIC_UNIVERSE,
    CsvProvider,
    DataError,
    SyntheticProvider,
    YahooFinanceProvider,
    save_csv,
    to_returns,
)
from asset_allocation_optimizer.optimization import OptimizationConstraints, max_sharpe, min_variance, risk_parity
from asset_allocation_optimizer.portfolio import equal_weights, portfolio_returns
from asset_allocation_optimizer.risk import risk_report
from asset_allocation_optimizer.simulation import monte_carlo

SAMPLE_FILE = Path(__file__).resolve().parents[2] / "data" / "sample_prices.csv"
SAMPLE_TICKERS = ["US_EQUITY", "INTL_EQUITY", "EM_EQUITY", "TECH", "BONDS", "GOLD", "REAL_ESTATE"]
WIDTH = 78


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asset-allocation-optimizer",
        description="Portfolio analytics, risk metrics and mean-variance optimization",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    analyze = subcommands.add_parser("analyze", help="Analyze and optimize a portfolio")
    analyze.add_argument("--source", choices=["sample", "yahoo", "csv"], default="sample")
    analyze.add_argument("--file", type=Path, help="CSV file when --source csv")
    analyze.add_argument("--tickers", nargs="+", help="Tickers to include")
    analyze.add_argument("--benchmark", help="Benchmark ticker")
    analyze.add_argument("--years", type=int, default=5, help="Years of history")
    analyze.add_argument("--risk-free", type=float, default=0.02, help="Annual risk-free rate")
    analyze.add_argument("--max-weight", type=float, default=1.0, help="Maximum weight per asset")
    analyze.add_argument("--cost", type=float, default=0.001, help="Transaction cost per trade")

    generate = subcommands.add_parser("generate-sample", help="Regenerate the synthetic sample dataset")
    generate.add_argument("--output", type=Path, default=SAMPLE_FILE)
    generate.add_argument("--seed", type=int, default=38)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "generate-sample":
            return _generate_sample(args)
        return _analyze(args)
    except (DataError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


def _generate_sample(args: argparse.Namespace) -> int:
    tickers = [profile.ticker for profile in DEFAULT_SYNTHETIC_UNIVERSE]
    prices = SyntheticProvider(DEFAULT_SYNTHETIC_UNIVERSE, seed=args.seed).fetch(
        tickers, date(2019, 1, 1), date(2025, 12, 31)
    )
    save_csv(prices, args.output)
    print(f"Saved {len(prices)} rows for {len(tickers)} assets to {args.output}")
    return 0


def _analyze(args: argparse.Namespace) -> int:
    tickers, benchmark = _resolve_universe(args)
    end = date.today() if args.source == "yahoo" else date(2025, 12, 31)
    start = end - timedelta(days=365 * args.years)
    provider = _provider(args)

    universe = tickers + ([benchmark] if benchmark and benchmark not in tickers else [])
    prices = provider.fetch(universe, start, end)
    returns = to_returns(prices)
    assets = returns[tickers]
    benchmark_returns = returns[benchmark] if benchmark else None

    constraints = OptimizationConstraints(max_weight=args.max_weight)
    portfolios = {"Equal Weight": equal_weights(tickers)}
    for optimizer in (min_variance, max_sharpe, risk_parity):
        result = optimizer(assets, args.risk_free, constraints)
        portfolios[result.name] = result.weights

    period = f"{prices.index[0].date()} -> {prices.index[-1].date()}"
    _header(f"Asset Allocation Optimizer  |  {period}  |  {len(prices)} days")
    _print_weights(portfolios)
    _print_metrics(assets, portfolios, benchmark_returns, args.risk_free, benchmark)
    _print_rebalancing(prices[tickers], portfolios["Maximum Sharpe"], args.cost)
    _print_simulation(portfolio_returns(assets, portfolios["Maximum Sharpe"]))
    return 0


def _resolve_universe(args: argparse.Namespace) -> tuple[list[str], str | None]:
    if args.source == "sample":
        return args.tickers or SAMPLE_TICKERS, args.benchmark or "MARKET"
    if not args.tickers:
        raise ValueError("--tickers is required for yahoo and csv sources")
    return args.tickers, args.benchmark


def _provider(args: argparse.Namespace):
    if args.source == "yahoo":
        return YahooFinanceProvider()
    if args.source == "csv":
        if args.file is None:
            raise ValueError("--file is required when --source csv")
        return CsvProvider(args.file)
    return CsvProvider(SAMPLE_FILE)


def _print_weights(portfolios: dict[str, pd.Series]) -> None:
    table = pd.DataFrame(portfolios)
    print("\n  Allocations")
    print(_indent(table.map(lambda w: f"{w:6.1%}").to_string()))


def _print_metrics(
    assets: pd.DataFrame,
    portfolios: dict[str, pd.Series],
    benchmark: pd.Series | None,
    risk_free: float,
    benchmark_name: str | None,
) -> None:
    rows = {}
    for name, weights in portfolios.items():
        report = risk_report(portfolio_returns(assets, weights), benchmark, risk_free)
        rows[name] = {
            "CAGR": f"{report.annual_return:+.2%}",
            "Volatility": f"{report.annual_volatility:.2%}",
            "Sharpe": f"{report.sharpe:.2f}",
            "Sortino": f"{report.sortino:.2f}",
            "Max DD": f"{-report.max_drawdown:.2%}",
            "VaR 95%": f"{report.var_95:.2%}",
            "CVaR 95%": f"{report.cvar_95:.2%}",
            **({"Beta": f"{report.beta:.2f}"} if report.beta is not None else {}),
        }
    if benchmark is not None:
        report = risk_report(benchmark, risk_free=risk_free)
        rows[f"Benchmark ({benchmark_name})"] = {
            "CAGR": f"{report.annual_return:+.2%}",
            "Volatility": f"{report.annual_volatility:.2%}",
            "Sharpe": f"{report.sharpe:.2f}",
            "Sortino": f"{report.sortino:.2f}",
            "Max DD": f"{-report.max_drawdown:.2%}",
            "VaR 95%": f"{report.var_95:.2%}",
            "CVaR 95%": f"{report.cvar_95:.2%}",
            "Beta": "1.00",
        }
    print("\n  Risk and return (in-sample)")
    print(_indent(pd.DataFrame(rows).T.to_string()))


def _print_rebalancing(prices: pd.DataFrame, weights: pd.Series, cost: float) -> None:
    rows = {}
    for frequency, result in compare_frequencies(prices, weights, transaction_cost=cost).items():
        rows[frequency.value.capitalize()] = {
            "Final value": f"${result.equity.iloc[-1]:,.0f}",
            "Return": f"{result.total_return:+.2%}",
            "Rebalances": len(result.rebalance_dates),
            "Costs": f"${result.total_costs:,.2f}",
        }
    print("\n  Rebalancing backtest of the Maximum Sharpe allocation ($10,000)")
    print(_indent(pd.DataFrame(rows).T.to_string()))


def _print_simulation(returns: pd.Series) -> None:
    result = monte_carlo(returns, horizon_days=252, simulations=5000)
    final = result.percentiles().iloc[-1]
    print("\n  Monte Carlo, 1 year ahead (5,000 bootstrap paths, $10,000)")
    print(f"    Pessimistic (p5)  ${final['p5']:>10,.0f}")
    print(f"    Median (p50)      ${final['p50']:>10,.0f}")
    print(f"    Optimistic (p95)  ${final['p95']:>10,.0f}")
    print(f"    Probability of loss  {result.probability_of_loss():.1%}")
    print("=" * WIDTH)


def _header(title: str) -> None:
    print("=" * WIDTH)
    print(f"  {title}")
    print("=" * WIDTH)


def _indent(text: str) -> str:
    return "\n".join("    " + line for line in text.splitlines())


if __name__ == "__main__":
    raise SystemExit(main())
