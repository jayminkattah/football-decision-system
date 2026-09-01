from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


ODDS_COLUMNS = ["home_odds", "draw_odds", "away_odds"]

RAW_IMPLIED_PROB_COLUMNS = [
    "raw_home_implied_prob",
    "raw_draw_implied_prob",
    "raw_away_implied_prob",
]

MARKET_PROB_COLUMNS = [
    "market_home_prob",
    "market_draw_prob",
    "market_away_prob",
]

MARKET_OVERROUND_COLUMN = "market_overround"


def validate_required_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """
    Validate that all required columns exist in the DataFrame.

    Parameters
    ----------
    df:
        Input DataFrame.
    required_columns:
        Columns required for processing.

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def add_market_implied_probabilities(
    df: pd.DataFrame,
    probability_sum_tolerance: float = 1e-6,
) -> pd.DataFrame:
    """
    Add raw and normalized market-implied probabilities from decimal odds.

    Input columns
    -------------
    home_odds:
        Decimal odds for home win.
    draw_odds:
        Decimal odds for draw.
    away_odds:
        Decimal odds for away win.

    Output columns
    --------------
    raw_home_implied_prob:
        1 / home_odds.
    raw_draw_implied_prob:
        1 / draw_odds.
    raw_away_implied_prob:
        1 / away_odds.
    market_overround:
        Sum of raw implied probabilities.
    market_home_prob:
        Normalized home probability after removing bookmaker margin.
    market_draw_prob:
        Normalized draw probability after removing bookmaker margin.
    market_away_prob:
        Normalized away probability after removing bookmaker margin.

    Notes
    -----
    This function does not train a model.
    This function does not calculate edge.
    This function only converts odds into normalized market probabilities.
    """
    validate_required_columns(df, ODDS_COLUMNS)

    result = df.copy()

    odds = result[ODDS_COLUMNS].apply(pd.to_numeric, errors="coerce")

    invalid_odds_mask = odds.isna().any(axis=1) | (odds <= 1.0).any(axis=1)

    if invalid_odds_mask.any():
        invalid_count = int(invalid_odds_mask.sum())
        raise ValueError(
            f"Found {invalid_count} rows with missing odds or decimal odds <= 1.0 "
            f"in required columns: {ODDS_COLUMNS}"
        )

    result[ODDS_COLUMNS] = odds

    result["raw_home_implied_prob"] = 1.0 / result["home_odds"]
    result["raw_draw_implied_prob"] = 1.0 / result["draw_odds"]
    result["raw_away_implied_prob"] = 1.0 / result["away_odds"]

    result[MARKET_OVERROUND_COLUMN] = result[RAW_IMPLIED_PROB_COLUMNS].sum(axis=1)

    invalid_overround_mask = result[MARKET_OVERROUND_COLUMN].isna() | (
        result[MARKET_OVERROUND_COLUMN] <= 0.0
    )

    if invalid_overround_mask.any():
        invalid_count = int(invalid_overround_mask.sum())
        raise ValueError(f"Found {invalid_count} rows with invalid market overround.")

    result["market_home_prob"] = (
        result["raw_home_implied_prob"] / result[MARKET_OVERROUND_COLUMN]
    )
    result["market_draw_prob"] = (
        result["raw_draw_implied_prob"] / result[MARKET_OVERROUND_COLUMN]
    )
    result["market_away_prob"] = (
        result["raw_away_implied_prob"] / result[MARKET_OVERROUND_COLUMN]
    )

    if len(result) > 0:
        probability_sums = result[MARKET_PROB_COLUMNS].sum(axis=1)

        if not np.allclose(
            probability_sums,
            1.0,
            atol=probability_sum_tolerance,
            rtol=0.0,
        ):
            max_error = float((probability_sums - 1.0).abs().max())
            raise ValueError(
                "Normalized market probabilities do not sum to 1.0 within tolerance. "
                f"Max absolute error: {max_error}"
            )

    return result


def build_market_probability_report(
    df: pd.DataFrame,
    probability_sum_tolerance: float = 1e-6,
) -> pd.DataFrame:
    """
    Build a simple evaluation report for market-implied probabilities.

    Input columns
    -------------
    market_overround
    market_home_prob
    market_draw_prob
    market_away_prob

    Output columns
    --------------
    metric
    value
    """
    required_columns = [MARKET_OVERROUND_COLUMN, *MARKET_PROB_COLUMNS]
    validate_required_columns(df, required_columns)

    if df.empty:
        report_rows = [
            ("n_rows", 0),
            ("mean_market_overround", np.nan),
            ("min_market_overround", np.nan),
            ("max_market_overround", np.nan),
            ("mean_market_home_prob", np.nan),
            ("mean_market_draw_prob", np.nan),
            ("mean_market_away_prob", np.nan),
            ("min_market_prob_sum", np.nan),
            ("max_market_prob_sum", np.nan),
            ("max_abs_market_prob_sum_error", np.nan),
            ("rows_prob_sum_not_close", 0),
        ]
        return pd.DataFrame(report_rows, columns=["metric", "value"])

    probability_sums = df[MARKET_PROB_COLUMNS].sum(axis=1)
    probability_sum_errors = (probability_sums - 1.0).abs()

    rows_prob_sum_not_close = int(
        (~np.isclose(
            probability_sums,
            1.0,
            atol=probability_sum_tolerance,
            rtol=0.0,
        )).sum()
    )

    report_rows = [
        ("n_rows", int(len(df))),
        ("mean_market_overround", float(df[MARKET_OVERROUND_COLUMN].mean())),
        ("min_market_overround", float(df[MARKET_OVERROUND_COLUMN].min())),
        ("max_market_overround", float(df[MARKET_OVERROUND_COLUMN].max())),
        ("mean_market_home_prob", float(df["market_home_prob"].mean())),
        ("mean_market_draw_prob", float(df["market_draw_prob"].mean())),
        ("mean_market_away_prob", float(df["market_away_prob"].mean())),
        ("min_market_prob_sum", float(probability_sums.min())),
        ("max_market_prob_sum", float(probability_sums.max())),
        ("max_abs_market_prob_sum_error", float(probability_sum_errors.max())),
        ("rows_prob_sum_not_close", rows_prob_sum_not_close),
    ]

    return pd.DataFrame(report_rows, columns=["metric", "value"])