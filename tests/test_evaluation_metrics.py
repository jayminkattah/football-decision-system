import numpy as np
import pandas as pd

from fbsystem.evaluation.metrics import (
    build_calibration_table,
    build_final_backtest_metrics,
    build_model_metrics,
    build_strategy_metrics,
)


def test_build_model_metrics_compares_model_and_market():
    matches = pd.DataFrame(
        {
            "actual_outcome": ["H", "D", "A", "H"],
            "pred_home_prob": [0.70, 0.30, 0.20, 0.60],
            "pred_draw_prob": [0.20, 0.40, 0.25, 0.25],
            "pred_away_prob": [0.10, 0.30, 0.55, 0.15],
            "market_home_prob": [0.60, 0.33, 0.25, 0.50],
            "market_draw_prob": [0.25, 0.34, 0.30, 0.30],
            "market_away_prob": [0.15, 0.33, 0.45, 0.20],
        }
    )

    metrics = build_model_metrics(matches)

    assert len(metrics) == 1
    assert metrics.loc[0, "n_matches"] == 4
    assert metrics.loc[0, "model_log_loss"] > 0
    assert metrics.loc[0, "model_brier_score"] > 0
    assert "model_log_loss_minus_market_log_loss" in metrics.columns


def test_build_calibration_table_returns_bins():
    matches = pd.DataFrame(
        {
            "actual_outcome": ["H", "D", "A", "H"],
            "pred_home_prob": [0.70, 0.30, 0.20, 0.60],
            "pred_draw_prob": [0.20, 0.40, 0.25, 0.25],
            "pred_away_prob": [0.10, 0.30, 0.55, 0.15],
        }
    )

    calibration = build_calibration_table(matches, n_bins=5)

    assert set(
        [
            "bin_left",
            "bin_right",
            "n_predictions",
            "mean_predicted_probability",
            "observed_rate",
            "absolute_calibration_error",
        ]
    ).issubset(calibration.columns)
    assert calibration["n_predictions"].sum() == 12


def test_build_strategy_metrics_summarizes_opportunities_and_bets():
    opportunities = pd.DataFrame(
        {
            "edge": [0.02, 0.04, 0.01],
            "predicted_probability": [0.55, 0.45, 0.35],
            "market_probability": [0.53, 0.41, 0.34],
        }
    )
    bets = pd.DataFrame(
        {
            "stake": [10.0, 10.0, 10.0],
            "total_return": [20.0, 0.0, 15.0],
            "odds": [2.0, 3.0, 1.5],
        }
    )

    metrics = build_strategy_metrics(opportunities, bets)

    assert metrics.loc[0, "n_strategy_opportunities"] == 3
    assert metrics.loc[0, "n_bets"] == 3
    assert metrics.loc[0, "total_staked"] == 30.0
    assert metrics.loc[0, "profit"] == 5.0
    assert np.isclose(metrics.loc[0, "roi"], 5.0 / 30.0)


def test_build_final_backtest_metrics_summarizes_stability():
    by_season = pd.DataFrame(
        {
            "season": ["2022-2023", "2023-2024"],
            "profit": [10.0, -5.0],
            "roi": [0.10, -0.05],
            "max_drawdown": [5.0, 12.0],
        }
    )
    cumulative = pd.DataFrame(
        {
            "cumulative_profit": [10.0, 5.0],
            "drawdown": [0.0, 5.0],
        }
    )

    metrics = build_final_backtest_metrics(by_season, cumulative)

    assert metrics.loc[0, "n_seasons"] == 2
    assert metrics.loc[0, "profitable_seasons"] == 1
    assert metrics.loc[0, "final_cumulative_profit"] == 5.0
    assert metrics.loc[0, "max_drawdown"] == 5.0
