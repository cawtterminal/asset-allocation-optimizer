from pathlib import Path

import pytest

from asset_allocation_optimizer.cli import main

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_cli_analyze_prints_full_report(capsys):
    assert main(["analyze", "--max-weight", "0.4"]) == 0
    output = capsys.readouterr().out
    for section in ["Allocations", "Risk and return", "Rebalancing backtest", "Monte Carlo"]:
        assert section in output
    assert "Maximum Sharpe" in output


def test_cli_reports_errors_without_traceback(capsys):
    assert main(["analyze", "--source", "yahoo"]) == 1
    assert "--tickers is required" in capsys.readouterr().err


def test_cli_generates_sample(tmp_path, capsys):
    output = tmp_path / "sample.csv"
    assert main(["generate-sample", "--output", str(output)]) == 0
    assert output.exists()


@pytest.mark.parametrize("portfolio", ["Equal Weight", "Maximum Sharpe", "Minimum Variance", "Risk Parity"])
def test_dashboard_renders_without_errors(portfolio):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(APP), default_timeout=120).run()
    app.segmented_control[0].set_value(portfolio).run()
    assert not app.exception
    assert not app.error
    assert [metric.label for metric in app.metric][:3] == ["CAGR", "Volatility", "Sharpe ratio"]
