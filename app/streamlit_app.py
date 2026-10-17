from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from asset_allocation_optimizer import charts
from asset_allocation_optimizer.backtest import RebalanceFrequency, compare_frequencies
from asset_allocation_optimizer.data import DataError, YahooFinanceProvider, clean_prices, load_csv, to_returns
from asset_allocation_optimizer.optimization import (
    OptimizationConstraints,
    OptimizationError,
    efficient_frontier,
    max_sharpe,
    min_variance,
    random_portfolios,
    risk_parity,
)
from asset_allocation_optimizer.portfolio import (
    covariance_matrix,
    equal_weights,
    expected_returns,
    normalize_weights,
    portfolio_returns,
    risk_contributions,
)
from asset_allocation_optimizer.risk import drawdown_series, risk_report
from asset_allocation_optimizer.simulation import monte_carlo

SAMPLE_FILE = Path(__file__).resolve().parents[1] / "data" / "sample_prices.csv"
SAMPLE_TICKERS = ["US_EQUITY", "INTL_EQUITY", "EM_EQUITY", "TECH", "BONDS", "GOLD", "REAL_ESTATE"]
DEFAULT_LIVE_TICKERS = "AAPL, MSFT, NVDA, JPM, XOM, GLD, TLT"
SOURCES = ["Sample dataset (synthetic)", "Yahoo Finance (live)", "Upload CSV"]

st.set_page_config(page_title="Asset Allocation Optimizer", page_icon="📈", layout="wide")


@st.cache_data(show_spinner=False)
def load_sample() -> pd.DataFrame:
    return load_csv(SAMPLE_FILE)


@st.cache_data(ttl=3600, show_spinner="Downloading prices from Yahoo Finance...")
def load_yahoo(tickers: tuple[str, ...], start: date, end: date) -> pd.DataFrame:
    return YahooFinanceProvider().fetch(list(tickers), start, end)


@st.cache_data(show_spinner=False)
def load_upload(content: bytes) -> pd.DataFrame:
    frame = pd.read_csv(pd.io.common.BytesIO(content))
    date_column = next((c for c in frame.columns if c.strip().lower() == "date"), None)
    if date_column is None:
        raise DataError("The CSV must have a 'Date' column")
    frame[date_column] = pd.to_datetime(frame[date_column])
    return clean_prices(frame.set_index(date_column).apply(pd.to_numeric, errors="coerce"))


def sidebar() -> dict:
    st.sidebar.title("Asset Allocation Optimizer")
    source = st.sidebar.radio("Data source", SOURCES)
    settings: dict = {"source": source}

    if source == SOURCES[0]:
        prices = load_sample()
        settings["tickers"] = st.sidebar.multiselect("Assets", SAMPLE_TICKERS, default=SAMPLE_TICKERS)
        settings["benchmark"] = "MARKET"
        settings["prices"] = prices
    elif source == SOURCES[1]:
        raw = st.sidebar.text_input("Tickers (comma separated)", DEFAULT_LIVE_TICKERS)
        settings["tickers"] = [t.strip().upper() for t in raw.split(",") if t.strip()]
        settings["benchmark"] = st.sidebar.text_input("Benchmark", "SPY").strip().upper() or None
        years = st.sidebar.slider("Years of history", 1, 20, 5)
        end = date.today()
        settings["live_range"] = (end - timedelta(days=365 * years), end)
    else:
        upload = st.sidebar.file_uploader("CSV with Date + one column per asset", type="csv")
        settings["upload"] = upload
        if upload is not None:
            prices = load_upload(upload.getvalue())
            settings["prices"] = prices
            columns = list(prices.columns)
            settings["benchmark"] = st.sidebar.selectbox("Benchmark", ["(none)", *columns])
            if settings["benchmark"] == "(none)":
                settings["benchmark"] = None
            options = [c for c in columns if c != settings["benchmark"]]
            settings["tickers"] = st.sidebar.multiselect("Assets", options, default=options)

    st.sidebar.divider()
    st.sidebar.subheader("Optimization")
    settings["risk_free"] = st.sidebar.number_input("Risk-free rate (annual)", 0.0, 0.2, 0.02, 0.005, "%.3f")
    settings["max_weight"] = st.sidebar.slider("Max weight per asset", 0.05, 1.0, 0.40, 0.05)

    st.sidebar.subheader("Backtest & simulation")
    settings["capital"] = st.sidebar.number_input("Initial capital", 1_000, 10_000_000, 10_000, 1_000)
    settings["cost"] = st.sidebar.number_input("Transaction cost", 0.0, 0.02, 0.001, 0.0005, "%.4f")
    settings["horizon"] = st.sidebar.slider("Simulation horizon (years)", 1, 10, 1)
    settings["simulations"] = st.sidebar.select_slider("Simulated paths", [1000, 2500, 5000, 10000], 5000)
    return settings


def resolve_prices(settings: dict) -> pd.DataFrame | None:
    tickers = settings.get("tickers") or []
    benchmark = settings.get("benchmark")
    if settings["source"] == SOURCES[1]:
        if not tickers:
            return None
        universe = tuple(dict.fromkeys([*tickers, *([benchmark] if benchmark else [])]))
        start, end = settings["live_range"]
        return load_yahoo(universe, start, end)
    prices = settings.get("prices")
    if prices is None:
        return None
    columns = list(dict.fromkeys([*tickers, *([benchmark] if benchmark else [])]))
    return prices[columns]


def build_portfolios(assets: pd.DataFrame, settings: dict) -> dict:
    constraints = OptimizationConstraints(max_weight=settings["max_weight"])
    optimal = [
        optimizer(assets, settings["risk_free"], constraints) for optimizer in (max_sharpe, min_variance, risk_parity)
    ]
    portfolios = {"Equal Weight": equal_weights(list(assets.columns))}
    portfolios.update({result.name: result.weights for result in optimal})
    return {"constraints": constraints, "optimal": optimal, "weights": portfolios}


def metrics_table(
    assets: pd.DataFrame, portfolios: dict[str, pd.Series], benchmark: pd.Series | None, risk_free: float
) -> pd.DataFrame:
    rows = {}
    for name, weights in portfolios.items():
        rows[name] = risk_report(portfolio_returns(assets, weights), benchmark, risk_free).to_dict()
    if benchmark is not None:
        rows[f"Benchmark ({benchmark.name})"] = risk_report(benchmark, risk_free=risk_free).to_dict()
    table = pd.DataFrame(rows).T.astype(float).dropna(axis=1, how="all")
    return table.rename(
        columns={
            "annual_return": "CAGR",
            "annual_volatility": "Volatility",
            "sharpe": "Sharpe",
            "sortino": "Sortino",
            "max_drawdown": "Max drawdown",
            "calmar": "Calmar",
            "var_95": "VaR 95% (1d)",
            "cvar_95": "CVaR 95% (1d)",
            "best_day": "Best day",
            "worst_day": "Worst day",
            "beta": "Beta",
            "alpha": "Alpha",
            "tracking_error": "Tracking error",
            "information_ratio": "Information ratio",
        }
    )


def format_metrics(table: pd.DataFrame):
    percent = [c for c in table.columns if c not in {"Sharpe", "Sortino", "Calmar", "Beta", "Information ratio"}]
    ratios = [c for c in table.columns if c not in percent]
    return table.style.format("{:.2%}", subset=percent, na_rep="—").format("{:.2f}", subset=ratios, na_rep="—")


def kpi_row(report, benchmark_report=None) -> None:
    columns = st.columns(5)
    delta = None
    if benchmark_report is not None:
        delta = f"{report.annual_return - benchmark_report.annual_return:+.2%} vs benchmark"
    columns[0].metric("CAGR", f"{report.annual_return:.2%}", delta)
    columns[1].metric("Volatility", f"{report.annual_volatility:.2%}")
    columns[2].metric("Sharpe ratio", f"{report.sharpe:.2f}")
    columns[3].metric("Max drawdown", f"{-report.max_drawdown:.2%}")
    columns[4].metric("VaR 95% (1 day)", f"{report.var_95:.2%}")


def custom_weights_editor(tickers: list[str], defaults: pd.Series) -> pd.Series:
    frame = pd.DataFrame({"Asset": tickers, "Weight %": (defaults.reindex(tickers).fillna(0) * 100).round(1)})
    edited = st.data_editor(
        frame,
        hide_index=True,
        width="stretch",
        disabled=["Asset"],
        column_config={"Weight %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=1.0)},
    )
    weights = pd.Series(edited["Weight %"].to_numpy(), index=edited["Asset"])
    return normalize_weights(weights)


def main() -> None:
    settings = sidebar()
    st.title("Asset Allocation Optimizer")
    st.caption("Portfolio analytics, risk metrics and mean-variance optimization")

    try:
        prices = resolve_prices(settings)
    except DataError as error:
        st.error(f"Could not load prices: {error}")
        st.info("Check the tickers or switch to the sample dataset in the sidebar.")
        return

    if prices is None:
        st.info("Choose a data source and at least two assets in the sidebar to start.")
        return
    if settings["source"] == SOURCES[0]:
        st.warning("The sample dataset is synthetic and generated for demonstration purposes.", icon="ℹ️")

    tickers = [t for t in settings["tickers"] if t in prices.columns]
    benchmark_name = settings.get("benchmark") if settings.get("benchmark") in prices.columns else None
    if len(tickers) < 2:
        st.info("Select at least two assets to run the optimization.")
        return

    returns = to_returns(prices)
    assets = returns[tickers]
    benchmark = returns[benchmark_name] if benchmark_name else None

    try:
        portfolios = build_portfolios(assets, settings)
    except (ValueError, OptimizationError) as error:
        st.error(f"Optimization failed: {error}")
        return

    st.caption(
        f"{prices.index[0].date()} → {prices.index[-1].date()} · {len(prices):,} trading days · "
        f"{len(tickers)} assets" + (f" · benchmark {benchmark_name}" if benchmark_name else "")
    )

    weights_by_name = portfolios["weights"]
    selected = st.segmented_control("Portfolio", [*weights_by_name, "Custom"], default="Maximum Sharpe")
    selected = selected or "Maximum Sharpe"
    if selected == "Custom":
        with st.expander("Custom weights", expanded=True):
            weights = custom_weights_editor(tickers, weights_by_name["Equal Weight"])
    else:
        weights = weights_by_name[selected]

    returns_selected = portfolio_returns(assets, weights)
    report = risk_report(returns_selected, benchmark, settings["risk_free"])
    benchmark_report = risk_report(benchmark, risk_free=settings["risk_free"]) if benchmark is not None else None
    kpi_row(report, benchmark_report)

    tabs = st.tabs(["Overview", "Risk", "Optimization", "Backtest", "Monte Carlo", "Data"])

    with tabs[0]:
        capital = settings["capital"]
        growth = pd.DataFrame({selected: (1 + returns_selected).cumprod() * capital})
        if benchmark is not None:
            growth[benchmark_name] = (1 + benchmark).cumprod() * capital
        st.plotly_chart(
            charts.growth_chart(
                growth,
                f"Growth of ${capital:,.0f}",
                charts.color_map(list(growth.columns), benchmark_name),
            ),
            width="stretch",
        )
        assets_growth = (1 + assets).cumprod() * capital
        st.plotly_chart(charts.growth_chart(assets_growth, "Individual assets"), width="stretch")

    with tabs[1]:
        left, right = st.columns(2)
        left.plotly_chart(charts.drawdown_chart(drawdown_series(returns_selected)), width="stretch")
        right.plotly_chart(charts.correlation_heatmap(assets.corr()), width="stretch")
        contributions = risk_contributions(weights, covariance_matrix(assets))
        st.plotly_chart(charts.risk_contribution_chart(weights, contributions), width="stretch")
        st.subheader("Risk and return by strategy")
        table = metrics_table(
            assets,
            weights_by_name | ({"Custom": weights} if selected == "Custom" else {}),
            benchmark,
            settings["risk_free"],
        )
        st.dataframe(format_metrics(table), width="stretch")

    with tabs[2]:
        constraints = portfolios["constraints"]
        cloud = random_portfolios(assets, 3000, settings["risk_free"])
        frontier = efficient_frontier(assets, 40, settings["risk_free"], constraints)
        asset_points = pd.DataFrame(
            {
                "return": expected_returns(assets),
                "volatility": assets.std() * (252**0.5),
            }
        )
        st.plotly_chart(
            charts.efficient_frontier_chart(cloud, frontier, portfolios["optimal"], asset_points),
            width="stretch",
        )
        st.caption(
            f"Long-only, maximum {settings['max_weight']:.0%} per asset. Random portfolios are "
            "unconstrained Dirichlet samples shown for context."
        )
        st.plotly_chart(charts.weights_chart(pd.DataFrame(weights_by_name)), width="stretch")

    with tabs[3]:
        results = compare_frequencies(prices[tickers], weights, settings["capital"], settings["cost"])
        equity = pd.DataFrame({f.value.capitalize(): r.equity for f, r in results.items()})
        st.plotly_chart(charts.growth_chart(equity, f"Rebalancing policies · {selected}"), width="stretch")
        summary = pd.DataFrame(
            {
                f.value.capitalize(): {
                    "Final value": r.equity.iloc[-1],
                    "Total return": r.total_return,
                    "Rebalances": len(r.rebalance_dates),
                    "Transaction costs": r.total_costs,
                    "Turnover": r.turnover,
                }
                for f, r in results.items()
            }
        ).T
        st.dataframe(
            summary.style.format(
                {
                    "Final value": "${:,.0f}",
                    "Total return": "{:+.2%}",
                    "Rebalances": "{:.0f}",
                    "Transaction costs": "${:,.2f}",
                    "Turnover": "{:.2f}x",
                }
            ),
            width="stretch",
        )
        best = summary["Final value"].idxmax()
        st.caption(
            f"Best policy in this period: {best}. Rebalancing frequency is evaluated with the "
            f"same target weights and a {settings['cost']:.2%} cost per trade."
        )
        if RebalanceFrequency.NEVER in results:
            drift = results[RebalanceFrequency.NEVER].weights.iloc[-1]
            st.caption(
                "Without rebalancing the allocation drifts to: " + ", ".join(f"{t} {w:.0%}" for t, w in drift.items())
            )

    with tabs[4]:
        horizon_days = settings["horizon"] * 252
        method = st.radio(
            "Method",
            ["bootstrap", "gbm"],
            horizontal=True,
            format_func=lambda m: "Historical bootstrap" if m == "bootstrap" else "Geometric Brownian motion",
        )
        simulation = monte_carlo(returns_selected, horizon_days, settings["simulations"], settings["capital"], method)
        final = simulation.percentiles().iloc[-1]
        columns = st.columns(4)
        columns[0].metric("Median outcome", f"${final['p50']:,.0f}")
        columns[1].metric("Pessimistic (p5)", f"${final['p5']:,.0f}")
        columns[2].metric("Optimistic (p95)", f"${final['p95']:,.0f}")
        columns[3].metric("Probability of loss", f"{simulation.probability_of_loss():.1%}")
        left, right = st.columns([3, 2])
        left.plotly_chart(charts.monte_carlo_chart(simulation), width="stretch")
        right.plotly_chart(charts.distribution_chart(simulation.final_values, settings["capital"]), width="stretch")

    with tabs[5]:
        st.dataframe(prices.sort_index(ascending=False), width="stretch")
        st.download_button("Download prices (CSV)", prices.to_csv().encode(), "prices.csv", "text/csv")
        weights_frame = pd.DataFrame(weights_by_name)
        st.download_button("Download allocations (CSV)", weights_frame.to_csv().encode(), "allocations.csv", "text/csv")

    st.divider()
    st.caption("Educational project. Past performance does not guarantee future results. Not investment advice.")


main()
