from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


MARKET_PROB_COLUMNS = [
    "market_home_prob",
    "market_draw_prob",
    "market_away_prob",
]

MODEL_PROB_COLUMNS = [
    "model_home_prob",
    "model_draw_prob",
    "model_away_prob",
]

EDGE_COLUMNS = [
    "home_edge",
    "draw_edge",
    "away_edge",
]

REQUIRED_EDGE_INPUT_COLUMNS = [
    *MARKET_PROB_COLUMNS,
    *MODEL_PROB_COLUMNS,
]


def validate_required_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def add_edges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add model-vs-market edge columns.

    Input columns
    -------------
    market_home_prob
    market_draw_prob
    market_away_prob
    model_home_prob
    model_draw_prob
    model_away_prob

    Output columns
    --------------
    home_edge
    draw_edge
    away_edge

    Notes
    -----
    Positive edge means the model probability is higher than the market-implied
    probability.

    This function does not select bets.
    This function does not apply staking.
    This function only calculates probability differences.
    """
    validate_required_columns(df, REQUIRED_EDGE_INPUT_COLUMNS)

    result = df.copy()

    result["home_edge"] = result["model_home_prob"] - result["market_home_prob"]
    result["draw_edge"] = result["model_draw_prob"] - result["market_draw_prob"]
    result["away_edge"] = result["model_away_prob"] - result["market_away_prob"]

    return result


def build_edge_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a simple report for edge columns.
    """
    validate_required_columns(df, EDGE_COLUMNS)

    if df.empty:
        rows = [
            ("n_rows", 0),
            ("mean_home_edge", np.nan),
            ("mean_draw_edge", np.nan),
            ("mean_away_edge", np.nan),
            ("max_home_edge", np.nan),
            ("max_draw_edge", np.nan),
            ("max_away_edge", np.nan),
            ("positive_home_edge_rows", 0),
            ("positive_draw_edge_rows", 0),
            ("positive_away_edge_rows", 0),
        ]
        return pd.DataFrame(rows, columns=["metric", "value"])

    rows = [
        ("n_rows", int(len(df))),
        ("mean_home_edge", float(df["home_edge"].mean())),
        ("mean_draw_edge", float(df["draw_edge"].mean())),
        ("mean_away_edge", float(df["away_edge"].mean())),
        ("min_home_edge", float(df["home_edge"].min())),
        ("min_draw_edge", float(df["draw_edge"].min())),
        ("min_away_edge", float(df["away_edge"].min())),
        ("max_home_edge", float(df["home_edge"].max())),
        ("max_draw_edge", float(df["draw_edge"].max())),
        ("max_away_edge", float(df["away_edge"].max())),
        ("positive_home_edge_rows", int((df["home_edge"] > 0).sum())),
        ("positive_draw_edge_rows", int((df["draw_edge"] > 0).sum())),
        ("positive_away_edge_rows", int((df["away_edge"] > 0).sum())),
    ]

    return pd.DataFrame(rows, columns=["metric", "value"])