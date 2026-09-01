import pandas as pd

from fbsystem.backtest.summary import (
    build_backtest_report,
    build_cumulative_backtest,
    calculate_max_drawdown,
    summarize_backtest_by_season,
)


def test_calculate_max_drawdown_uses_starting_zero_as_peak():
    profit = pd.Series([-10.0, 5.0, -20.0, 15.0])

    assert calculate_max_drawdown(profit) == 25.0


def test_summarize_backtest_by_season_returns_expected_metrics():
    bets = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2023-08-01",
                    "2023-08-08",
                    "2024-08-01",
                ]
            ),
            "season": ["2023-2024", "2023-2024", "2024-2025"],
            "stake": [10.0, 10.0, 20.0],
            "odds": [2.0, 3.0, 2.5],
            "total_return": [20.0, 0.0, 50.0],
        }
    )

    summary = summarize_backtest_by_season(bets)

    assert list(summary["season"]) == ["2023-2024", "2024-2025"]

    first = summary.loc[summary["season"] == "2023-2024"].iloc[0]
    assert first["n_bets"] == 2
    assert first["total_staked"] == 20.0
    assert first["total_return"] == 20.0
    assert first["profit"] == 0.0
    assert first["roi"] == 0.0
    assert first["win_rate"] == 0.5


def test_build_cumulative_backtest_has_running_profit_and_roi():
    bets = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2023-08-01",
                    "2023-08-08",
                    "2023-08-15",
                ]
            ),
            "season": ["2023-2024", "2023-2024", "2023-2024"],
            "stake": [10.0, 10.0, 10.0],
            "total_return": [20.0, 0.0, 30.0],
        }
    )

    cumulative = build_cumulative_backtest(bets)

    assert list(cumulative["bet_number"]) == [1, 2, 3]
    assert list(cumulative["profit"]) == [10.0, -10.0, 20.0]
    assert cumulative["cumulative_profit"].iloc[-1] == 20.0
    assert cumulative["running_roi"].iloc[-1] == 20.0 / 30.0


def test_build_backtest_report_returns_single_row():
    bets = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-08-01", "2023-08-08"]),
            "season": ["2023-2024", "2023-2024"],
            "stake": [10.0, 10.0],
            "total_return": [20.0, 0.0],
        }
    )

    report = build_backtest_report(bets)

    assert len(report) == 1
    assert report.loc[0, "n_bets"] == 2
    assert report.loc[0, "total_staked"] == 20.0
    assert report.loc[0, "profit"] == 0.0
