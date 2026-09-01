import pandas as pd
import pytest

from fbsystem.features.edges import (
    EDGE_COLUMNS,
    add_edges,
    build_edge_report,
)


def make_edge_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_home_prob": [0.50, 0.30],
            "market_draw_prob": [0.25, 0.30],
            "market_away_prob": [0.25, 0.40],
            "model_home_prob": [0.55, 0.25],
            "model_draw_prob": [0.20, 0.35],
            "model_away_prob": [0.25, 0.40],
        }
    )


def test_add_edges_adds_expected_columns():
    df = make_edge_df()

    result = add_edges(df)

    for column in EDGE_COLUMNS:
        assert column in result.columns


def test_add_edges_calculates_expected_values():
    df = make_edge_df()

    result = add_edges(df)

    assert result.loc[0, "home_edge"] == pytest.approx(0.05)
    assert result.loc[0, "draw_edge"] == pytest.approx(-0.05)
    assert result.loc[0, "away_edge"] == pytest.approx(0.00)

    assert result.loc[1, "home_edge"] == pytest.approx(-0.05)
    assert result.loc[1, "draw_edge"] == pytest.approx(0.05)
    assert result.loc[1, "away_edge"] == pytest.approx(0.00)


def test_add_edges_does_not_mutate_input_dataframe():
    df = make_edge_df()
    original_columns = df.columns.tolist()

    _ = add_edges(df)

    assert df.columns.tolist() == original_columns


def test_add_edges_raises_for_missing_columns():
    df = make_edge_df().drop(columns=["model_home_prob"])

    with pytest.raises(ValueError, match="Missing required columns"):
        add_edges(df)


def test_build_edge_report_returns_expected_metrics():
    df = add_edges(make_edge_df())

    report = build_edge_report(df)

    assert list(report.columns) == ["metric", "value"]

    metrics = set(report["metric"])

    expected_metrics = {
        "n_rows",
        "mean_home_edge",
        "mean_draw_edge",
        "mean_away_edge",
        "min_home_edge",
        "min_draw_edge",
        "min_away_edge",
        "max_home_edge",
        "max_draw_edge",
        "max_away_edge",
        "positive_home_edge_rows",
        "positive_draw_edge_rows",
        "positive_away_edge_rows",
    }

    assert expected_metrics.issubset(metrics)

    report_lookup = dict(zip(report["metric"], report["value"]))

    assert report_lookup["n_rows"] == 2
    assert report_lookup["positive_home_edge_rows"] == 1
    assert report_lookup["positive_draw_edge_rows"] == 1
    assert report_lookup["positive_away_edge_rows"] == 0