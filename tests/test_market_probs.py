import numpy as np
import pandas as pd
import pytest

from fbsystem.features.market_probs import (
    MARKET_PROB_COLUMNS,
    RAW_IMPLIED_PROB_COLUMNS,
    add_market_implied_probabilities,
    build_market_probability_report,
)


def test_add_market_implied_probabilities_adds_expected_columns():
    df = pd.DataFrame(
        {
            "match_id": ["match_1"],
            "home_odds": [2.0],
            "draw_odds": [4.0],
            "away_odds": [4.0],
        }
    )

    result = add_market_implied_probabilities(df)

    expected_new_columns = [
        "raw_home_implied_prob",
        "raw_draw_implied_prob",
        "raw_away_implied_prob",
        "market_overround",
        "market_home_prob",
        "market_draw_prob",
        "market_away_prob",
    ]

    for column in expected_new_columns:
        assert column in result.columns


def test_add_market_implied_probabilities_calculates_raw_probs_and_overround():
    df = pd.DataFrame(
        {
            "home_odds": [2.0],
            "draw_odds": [4.0],
            "away_odds": [4.0],
        }
    )

    result = add_market_implied_probabilities(df)

    assert result.loc[0, "raw_home_implied_prob"] == pytest.approx(0.50)
    assert result.loc[0, "raw_draw_implied_prob"] == pytest.approx(0.25)
    assert result.loc[0, "raw_away_implied_prob"] == pytest.approx(0.25)
    assert result.loc[0, "market_overround"] == pytest.approx(1.00)


def test_add_market_implied_probabilities_normalizes_probabilities_to_one():
    df = pd.DataFrame(
        {
            "home_odds": [1.80],
            "draw_odds": [3.60],
            "away_odds": [5.00],
        }
    )

    result = add_market_implied_probabilities(df)

    probability_sum = result[MARKET_PROB_COLUMNS].sum(axis=1).iloc[0]

    assert probability_sum == pytest.approx(1.0)


def test_add_market_implied_probabilities_uses_bookmaker_margin_normalization():
    df = pd.DataFrame(
        {
            "home_odds": [2.0],
            "draw_odds": [3.0],
            "away_odds": [4.0],
        }
    )

    result = add_market_implied_probabilities(df)

    raw_home = 1 / 2.0
    raw_draw = 1 / 3.0
    raw_away = 1 / 4.0
    overround = raw_home + raw_draw + raw_away

    assert result.loc[0, "market_home_prob"] == pytest.approx(raw_home / overround)
    assert result.loc[0, "market_draw_prob"] == pytest.approx(raw_draw / overround)
    assert result.loc[0, "market_away_prob"] == pytest.approx(raw_away / overround)


def test_add_market_implied_probabilities_does_not_mutate_input_dataframe():
    df = pd.DataFrame(
        {
            "home_odds": [2.0],
            "draw_odds": [3.0],
            "away_odds": [4.0],
        }
    )

    original_columns = df.columns.tolist()

    _ = add_market_implied_probabilities(df)

    assert df.columns.tolist() == original_columns


def test_add_market_implied_probabilities_raises_for_missing_columns():
    df = pd.DataFrame(
        {
            "home_odds": [2.0],
            "draw_odds": [3.0],
        }
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        add_market_implied_probabilities(df)


def test_add_market_implied_probabilities_raises_for_invalid_odds():
    df = pd.DataFrame(
        {
            "home_odds": [2.0],
            "draw_odds": [1.0],
            "away_odds": [4.0],
        }
    )

    with pytest.raises(ValueError, match="decimal odds <= 1.0"):
        add_market_implied_probabilities(df)


def test_add_market_implied_probabilities_raises_for_missing_odds():
    df = pd.DataFrame(
        {
            "home_odds": [2.0],
            "draw_odds": [np.nan],
            "away_odds": [4.0],
        }
    )

    with pytest.raises(ValueError, match="missing odds"):
        add_market_implied_probabilities(df)


def test_add_market_implied_probabilities_handles_empty_dataframe():
    df = pd.DataFrame(
        columns=[
            "home_odds",
            "draw_odds",
            "away_odds",
        ]
    )

    result = add_market_implied_probabilities(df)

    assert result.empty

    for column in [*RAW_IMPLIED_PROB_COLUMNS, "market_overround", *MARKET_PROB_COLUMNS]:
        assert column in result.columns


def test_build_market_probability_report_returns_expected_metrics():
    df = pd.DataFrame(
        {
            "home_odds": [2.0, 1.80],
            "draw_odds": [3.0, 3.60],
            "away_odds": [4.0, 5.00],
        }
    )

    result = add_market_implied_probabilities(df)
    report = build_market_probability_report(result)

    assert list(report.columns) == ["metric", "value"]

    metrics = set(report["metric"])

    expected_metrics = {
        "n_rows",
        "mean_market_overround",
        "min_market_overround",
        "max_market_overround",
        "mean_market_home_prob",
        "mean_market_draw_prob",
        "mean_market_away_prob",
        "min_market_prob_sum",
        "max_market_prob_sum",
        "max_abs_market_prob_sum_error",
        "rows_prob_sum_not_close",
    }

    assert expected_metrics.issubset(metrics)

    report_lookup = dict(zip(report["metric"], report["value"]))

    assert report_lookup["n_rows"] == 2
    assert report_lookup["rows_prob_sum_not_close"] == 0
    assert report_lookup["max_abs_market_prob_sum_error"] == pytest.approx(0.0)