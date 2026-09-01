from __future__ import annotations

from pathlib import Path

import pandas as pd

from fbsystem.evaluation.metrics import (
    build_calibration_table,
    build_final_backtest_metrics,
    build_model_metrics,
    build_strategy_metrics,
    save_cumulative_profit_figure,
    save_model_calibration_figure,
    save_roi_by_season_figure,
)


def main() -> None:
    matches_path = Path("data/processed/matches_with_predictions.parquet")
    strategy_path = Path("data/processed/strategy_opportunities.parquet")
    bets_path = Path("data/processed/simulated_bets.parquet")
    backtest_by_season_path = Path("outputs/backtests/backtest_by_season.csv")
    backtest_cumulative_path = Path("outputs/backtests/backtest_cumulative.csv")

    required_paths = [
        matches_path,
        strategy_path,
        bets_path,
        backtest_by_season_path,
        backtest_cumulative_path,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    evaluation_dir = Path("outputs/evaluation")
    figures_dir = Path("outputs/figures")

    evaluation_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    matches = pd.read_parquet(matches_path)
    strategy_opportunities = pd.read_parquet(strategy_path)
    simulated_bets = pd.read_parquet(bets_path)
    backtest_by_season = pd.read_csv(backtest_by_season_path)
    backtest_cumulative = pd.read_csv(backtest_cumulative_path)

    model_metrics = build_model_metrics(matches)
    strategy_metrics = build_strategy_metrics(strategy_opportunities, simulated_bets)
    backtest_metrics = build_final_backtest_metrics(backtest_by_season, backtest_cumulative)
    calibration_table = build_calibration_table(matches)

    final_model_metrics_path = evaluation_dir / "final_model_metrics.csv"
    final_strategy_metrics_path = evaluation_dir / "final_strategy_metrics.csv"
    final_backtest_metrics_path = evaluation_dir / "final_backtest_metrics.csv"

    model_metrics.to_csv(final_model_metrics_path, index=False)
    strategy_metrics.to_csv(final_strategy_metrics_path, index=False)
    backtest_metrics.to_csv(final_backtest_metrics_path, index=False)

    cumulative_profit_path = figures_dir / "cumulative_profit.png"
    roi_by_season_path = figures_dir / "roi_by_season.png"
    calibration_path = figures_dir / "model_calibration_basic.png"

    save_cumulative_profit_figure(backtest_cumulative, cumulative_profit_path)
    save_roi_by_season_figure(backtest_by_season, roi_by_season_path)
    save_model_calibration_figure(calibration_table, calibration_path)

    print("Evaluation outputs build complete.")
    print(f"Model metrics: {final_model_metrics_path}")
    print(f"Strategy metrics: {final_strategy_metrics_path}")
    print(f"Backtest metrics: {final_backtest_metrics_path}")
    print(f"Cumulative profit figure: {cumulative_profit_path}")
    print(f"ROI by season figure: {roi_by_season_path}")
    print(f"Calibration figure: {calibration_path}")


if __name__ == "__main__":
    main()
