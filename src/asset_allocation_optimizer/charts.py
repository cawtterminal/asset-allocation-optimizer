from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from asset_allocation_optimizer.optimization import OptimalPortfolio
from asset_allocation_optimizer.simulation import SimulationResult

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE_RAMP = {200: "#9ec5f4", 300: "#6da7ec", 450: "#2a78d6", 550: "#1c5cab"}
DIVERGING = [[0.0, "#2a78d6"], [0.25, "#86b6ef"], [0.5, "#f0efec"], [0.75, "#ee8f8e"], [1.0, "#e34948"]]
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


STRATEGY_COLORS = {
    "Equal Weight": SERIES[0],
    "Maximum Sharpe": SERIES[1],
    "Minimum Variance": SERIES[2],
    "Risk Parity": SERIES[3],
    "Custom": SERIES[4],
}


def color_map(names: list[str], benchmark: str | None = None) -> dict[str, str]:
    colors = {}
    free_slots = iter(SERIES)
    for name in names:
        if name == benchmark:
            colors[name] = INK_MUTED
        elif name in STRATEGY_COLORS:
            colors[name] = STRATEGY_COLORS[name]
        else:
            colors[name] = next(free_slots, INK_MUTED)
    return colors


def growth_chart(series: pd.DataFrame, title: str, colors: dict[str, str] | None = None) -> go.Figure:
    colors = colors or color_map(list(series.columns))
    figure = go.Figure()
    for name in series.columns:
        figure.add_trace(
            go.Scatter(
                x=series.index,
                y=series[name],
                name=name,
                mode="lines",
                line={"width": 2, "color": colors[name]},
                hovertemplate="%{y:$,.0f}<extra>" + name + "</extra>",
            )
        )
    _style(figure, title, y_format="$,.0f")
    figure.update_layout(hovermode="x unified")
    return figure


def drawdown_chart(drawdown: pd.Series, title: str = "Drawdown") -> go.Figure:
    figure = go.Figure(
        go.Scatter(
            x=drawdown.index,
            y=drawdown,
            mode="lines",
            name="Drawdown",
            line={"width": 2, "color": SERIES[7]},
            fill="tozeroy",
            fillcolor="rgba(227, 73, 72, 0.12)",
            hovertemplate="%{y:.2%}<extra></extra>",
        )
    )
    _style(figure, title, y_format=".0%", legend=False)
    figure.update_layout(hovermode="x unified")
    return figure


def correlation_heatmap(correlation: pd.DataFrame) -> go.Figure:
    labels = list(correlation.columns)
    figure = go.Figure(
        go.Heatmap(
            z=correlation.to_numpy(),
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale=DIVERGING,
            xgap=2,
            ygap=2,
            text=np.round(correlation.to_numpy(), 2),
            texttemplate="%{text:.2f}",
            textfont={"color": INK, "size": 12},
            hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
            colorbar={"title": {"text": "ρ"}, "thickness": 12, "outlinewidth": 0},
        )
    )
    _style(figure, "Correlation matrix", legend=False)
    figure.update_xaxes(showgrid=False, side="bottom")
    figure.update_yaxes(showgrid=False, autorange="reversed")
    return figure


def efficient_frontier_chart(
    cloud: pd.DataFrame,
    frontier: pd.DataFrame,
    highlights: list[OptimalPortfolio],
    assets: pd.DataFrame,
) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=cloud["volatility"],
            y=cloud["return"],
            mode="markers",
            name="Random portfolios",
            marker={"size": 5, "color": BASELINE, "opacity": 0.55},
            customdata=cloud["sharpe"],
            hovertemplate="Vol %{x:.1%}<br>Return %{y:.1%}<br>Sharpe %{customdata:.2f}<extra></extra>",
        )
    )
    if not frontier.empty:
        figure.add_trace(
            go.Scatter(
                x=frontier["volatility"],
                y=frontier["return"],
                mode="lines",
                name="Efficient frontier",
                line={"width": 2, "color": INK},
                hovertemplate="Vol %{x:.1%}<br>Return %{y:.1%}<extra>Frontier</extra>",
            )
        )
    figure.add_trace(
        go.Scatter(
            x=assets["volatility"],
            y=assets["return"],
            mode="markers+text",
            name="Individual assets",
            text=assets.index,
            textposition="top center",
            textfont={"color": INK_SECONDARY, "size": 11},
            cliponaxis=False,
            marker={"size": 9, "symbol": "diamond", "color": INK_MUTED, "line": {"width": 2, "color": SURFACE}},
            hovertemplate="%{text}<br>Vol %{x:.1%}<br>Return %{y:.1%}<extra></extra>",
        )
    )
    symbols = ["star", "circle", "square", "triangle-up"]
    for i, portfolio in enumerate(highlights):
        figure.add_trace(
            go.Scatter(
                x=[portfolio.volatility],
                y=[portfolio.expected_return],
                mode="markers",
                name=portfolio.name,
                marker={
                    "size": 15,
                    "symbol": symbols[i % len(symbols)],
                    "color": STRATEGY_COLORS.get(portfolio.name, SERIES[i % len(SERIES)]),
                    "line": {"width": 2, "color": SURFACE},
                },
                hovertemplate=(
                    f"{portfolio.name}<br>Vol %{{x:.1%}}<br>Return %{{y:.1%}}"
                    f"<br>Sharpe {portfolio.sharpe:.2f}<extra></extra>"
                ),
            )
        )
    _style(figure, "Efficient frontier", y_format=".0%", x_format=".0%")
    figure.update_xaxes(title_text="Annual volatility")
    figure.update_yaxes(title_text="Expected annual return")
    figure.update_layout(margin={"l": 10, "r": 10, "t": 70, "b": 10})
    return figure


def weights_chart(allocations: pd.DataFrame) -> go.Figure:
    colors = color_map(list(allocations.columns))
    figure = go.Figure()
    for name in allocations.columns:
        figure.add_trace(
            go.Bar(
                y=allocations.index,
                x=allocations[name],
                name=name,
                orientation="h",
                marker={"color": colors[name], "line": {"width": 0}, "cornerradius": 4},
                hovertemplate="%{y}: %{x:.1%}<extra>" + name + "</extra>",
            )
        )
    _style(figure, "Allocation by strategy", x_format=".0%")
    figure.update_layout(barmode="group", bargap=0.3, bargroupgap=0.12)
    figure.update_yaxes(autorange="reversed", showgrid=False)
    return figure


def risk_contribution_chart(weights: pd.Series, contributions: pd.Series) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=weights.index,
            y=weights,
            name="Capital weight",
            marker={"color": SERIES[0], "cornerradius": 4},
            hovertemplate="%{x}: %{y:.1%}<extra>Capital</extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            x=contributions.index,
            y=contributions,
            name="Risk contribution",
            marker={"color": SERIES[1], "cornerradius": 4},
            hovertemplate="%{x}: %{y:.1%}<extra>Risk</extra>",
        )
    )
    _style(figure, "Capital vs risk allocation", y_format=".0%")
    figure.update_layout(barmode="group", bargap=0.3, bargroupgap=0.12)
    figure.update_xaxes(showgrid=False)
    return figure


def monte_carlo_chart(result: SimulationResult, sample_paths: int = 40) -> go.Figure:
    bands = result.percentiles()
    days = np.arange(len(bands))
    figure = go.Figure()
    for path in result.paths[:sample_paths]:
        figure.add_trace(
            go.Scatter(
                x=days,
                y=path,
                mode="lines",
                line={"width": 1, "color": BASELINE},
                opacity=0.35,
                hoverinfo="skip",
                showlegend=False,
            )
        )
    _band(figure, days, bands["p5"], bands["p95"], BLUE_RAMP[200], "5th-95th percentile")
    _band(figure, days, bands["p25"], bands["p75"], BLUE_RAMP[300], "25th-75th percentile")
    figure.add_trace(
        go.Scatter(
            x=days,
            y=bands["p50"],
            mode="lines",
            name="Median",
            line={"width": 2, "color": BLUE_RAMP[550]},
            hovertemplate="Day %{x}<br>Median %{y:$,.0f}<extra></extra>",
        )
    )
    figure.add_hline(y=result.initial_value, line={"width": 1, "dash": "dot", "color": INK_MUTED})
    _style(figure, "Simulated portfolio value", y_format="$,.0f")
    figure.update_xaxes(title_text="Trading days ahead")
    return figure


def distribution_chart(final_values: np.ndarray, initial_value: float) -> go.Figure:
    figure = go.Figure(
        go.Histogram(
            x=final_values,
            nbinsx=60,
            marker={"color": SERIES[0], "line": {"width": 1, "color": SURFACE}},
            hovertemplate="%{x:$,.0f}: %{y} paths<extra></extra>",
        )
    )
    figure.add_vline(x=initial_value, line={"width": 2, "dash": "dot", "color": INK})
    _style(figure, "Distribution of final values", x_format="$,.0f", legend=False)
    figure.update_yaxes(title_text="Paths")
    return figure


def _band(figure: go.Figure, x: np.ndarray, low: pd.Series, high: pd.Series, color: str, name: str) -> None:
    figure.add_trace(go.Scatter(x=x, y=high, mode="lines", line={"width": 0}, hoverinfo="skip", showlegend=False))
    figure.add_trace(
        go.Scatter(
            x=x,
            y=low,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor=color,
            name=name,
            opacity=0.6,
            hoverinfo="skip",
        )
    )


def _style(
    figure: go.Figure, title: str, y_format: str | None = None, x_format: str | None = None, legend: bool = True
) -> None:
    figure.update_layout(
        title={"text": title, "font": {"size": 16, "color": INK}, "x": 0, "xanchor": "left"},
        font={"family": FONT, "color": INK_SECONDARY, "size": 12},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        showlegend=legend,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.0,
            "xanchor": "right",
            "x": 1,
            "font": {"color": INK_SECONDARY},
        },
        hoverlabel={"bgcolor": SURFACE, "font": {"color": INK, "family": FONT}, "bordercolor": GRID},
    )
    axis = {
        "gridcolor": GRID,
        "zerolinecolor": BASELINE,
        "linecolor": BASELINE,
        "tickfont": {"color": INK_MUTED},
        "title_font": {"color": INK_SECONDARY},
    }
    figure.update_xaxes(**axis, tickformat=x_format)
    figure.update_yaxes(**axis, tickformat=y_format)
