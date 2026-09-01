# dashboard/utils/load_data.py

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[2]


ARTIFACT_PATHS: dict[str, str] = {
    "matches": "data/processed/matches_with_predictions.parquet",
    "opportunities": "data/processed/strategy_opportunities.parquet",
    "simulated_bets": "data/processed/simulated_bets.parquet",
    "backtest_by_season": "outputs/backtests/backtest_by_season.csv",
    "backtest_cumulative": "outputs/backtests/backtest_cumulative.csv",
    "model_metrics": "outputs/evaluation/final_model_metrics.csv",
    "strategy_metrics": "outputs/evaluation/final_strategy_metrics.csv",
    "backtest_metrics": "outputs/evaluation/final_backtest_metrics.csv",
    "cumulative_profit_figure": "outputs/figures/cumulative_profit.png",
    "roi_by_season_figure": "outputs/figures/roi_by_season.png",
    "model_calibration_figure": "outputs/figures/model_calibration_basic.png",
}


def project_path(relative_path: str) -> Path:
    return ROOT_DIR / relative_path


def artifact_status() -> pd.DataFrame:
    rows = []

    for name, relative_path in ARTIFACT_PATHS.items():
        path = project_path(relative_path)
        rows.append(
            {
                "artifact": name,
                "path": relative_path,
                "exists": path.exists(),
            }
        )

    return pd.DataFrame(rows)


def _read_parquet(relative_path: str) -> pd.DataFrame:
    path = project_path(relative_path)

    if not path.exists():
        return pd.DataFrame()

    return pd.read_parquet(path)


def _read_csv(relative_path: str) -> pd.DataFrame:
    path = project_path(relative_path)

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_dashboard_data() -> dict[str, pd.DataFrame]:
    return {
        "matches": _read_parquet(ARTIFACT_PATHS["matches"]),
        "opportunities": _read_parquet(ARTIFACT_PATHS["opportunities"]),
        "simulated_bets": _read_parquet(ARTIFACT_PATHS["simulated_bets"]),
        "backtest_by_season": _read_csv(ARTIFACT_PATHS["backtest_by_season"]),
        "backtest_cumulative": _read_csv(ARTIFACT_PATHS["backtest_cumulative"]),
        "model_metrics": _read_csv(ARTIFACT_PATHS["model_metrics"]),
        "strategy_metrics": _read_csv(ARTIFACT_PATHS["strategy_metrics"]),
        "backtest_metrics": _read_csv(ARTIFACT_PATHS["backtest_metrics"]),
    }


def metrics_to_dict(metrics_df: pd.DataFrame) -> dict[str, Any]:
    if metrics_df.empty:
        return {}

    columns = list(metrics_df.columns)

    if {"metric", "value"}.issubset(columns):
        return dict(zip(metrics_df["metric"], metrics_df["value"]))

    if len(columns) >= 2 and metrics_df.shape[1] == 2:
        return dict(zip(metrics_df.iloc[:, 0], metrics_df.iloc[:, 1]))

    if len(metrics_df) == 1:
        return metrics_df.iloc[0].to_dict()

    return {column: metrics_df[column].iloc[0] for column in columns}


def existing_figure_paths() -> dict[str, Path]:
    figure_keys = [
        "cumulative_profit_figure",
        "roi_by_season_figure",
        "model_calibration_figure",
    ]

    return {
        key: project_path(ARTIFACT_PATHS[key])
        for key in figure_keys
        if project_path(ARTIFACT_PATHS[key]).exists()
    }