import pandas as pd
import pytest

from fbsystem.models.baseline import MODEL_PROB_COLUMNS
from fbsystem.models.walk_forward import (
    WALK_FORWARD_METADATA_COLUMNS,
    build_predictions_report,
    generate_walk_forward_predictions,
    get_sorted_seasons,
)


def make_walk_forward_df() -> pd.DataFrame:
    rows = []

    seasons = [2021, 2022, 2023]

    for season in seasons:
        rows.extend(
            [
                {
                    "match_id": f"{season}_1",
                    "season_start_year": season,
                    "market_home_prob": 0.55,
                    "market_draw_prob": 0.25,
                    "market_away_prob": 0.20,
                    "source_country": "england",
                    "source_league_code": "E0",
                    "full_time_result": "H",
                },
                {
                    "match_id": f"{season}_2",
                    "season_start_year": season,
                    "market_home_prob": 0.30,
                    "market_draw_prob": 0.30,
                    "market_away_prob": 0.40,
                    "source_country": "spain",
                    "source_league_code": "SP1",
                    "full_time_result": "A",
                },
                {
                    "match_id": f"{season}_3",
                    "season_start_year": season,
                    "market_home_prob": 0.35,
                    "market_draw_prob": 0.35,
                    "market_away_prob": 0.30,
                    "source_country": "germany",
                    "source_league_code": "D1",
                    "full_time_result": "D",
                },
            ]
        )

    return pd.DataFrame(rows)


def test_get_sorted_seasons_returns_sorted_unique_seasons():
    df = pd.DataFrame(
        {
            "season_start_year": [2023, 2021, 2022, 2021],
        }
    )

    seasons = get_sorted_seasons(df)

    assert seasons == [2021, 2022, 2023]


def test_generate_walk_forward_predictions_skips_first_season():
    df = make_walk_forward_df()

    predictions = generate_walk_forward_predictions(df)

    predicted_seasons = sorted(predictions["prediction_test_season"].unique())

    assert predicted_seasons == [2022, 2023]
    assert 2021 not in predicted_seasons


def test_generate_walk_forward_predictions_adds_model_probability_columns():
    df = make_walk_forward_df()

    predictions = generate_walk_forward_predictions(df)

    for column in MODEL_PROB_COLUMNS:
        assert column in predictions.columns


def test_generate_walk_forward_predictions_adds_metadata_columns():
    df = make_walk_forward_df()

    predictions = generate_walk_forward_predictions(df)

    for column in WALK_FORWARD_METADATA_COLUMNS:
        assert column in predictions.columns


def test_generate_walk_forward_predictions_probabilities_sum_to_one():
    df = make_walk_forward_df()

    predictions = generate_walk_forward_predictions(df)

    probability_sums = predictions[MODEL_PROB_COLUMNS].sum(axis=1)

    assert probability_sums.min() == pytest.approx(1.0)
    assert probability_sums.max() == pytest.approx(1.0)


def test_generate_walk_forward_predictions_uses_only_past_seasons_for_training():
    df = make_walk_forward_df()

    predictions = generate_walk_forward_predictions(df)

    assert (
        predictions["prediction_train_end_season"]
        < predictions["prediction_test_season"]
    ).all()


def test_generate_walk_forward_predictions_returns_expected_row_count():
    df = make_walk_forward_df()

    predictions = generate_walk_forward_predictions(df)

    # First season is skipped. Seasons 2022 and 2023 each have 3 rows.
    assert len(predictions) == 6


def test_generate_walk_forward_predictions_raises_for_missing_columns():
    df = make_walk_forward_df().drop(columns=["market_home_prob"])

    with pytest.raises(ValueError, match="Missing required columns"):
        generate_walk_forward_predictions(df)


def test_generate_walk_forward_predictions_returns_empty_when_only_one_season():
    df = make_walk_forward_df()
    df = df[df["season_start_year"] == 2021]

    predictions = generate_walk_forward_predictions(df)

    assert predictions.empty


def test_build_predictions_report_returns_expected_metrics():
    df = make_walk_forward_df()
    predictions = generate_walk_forward_predictions(df)

    report = build_predictions_report(predictions)

    assert list(report.columns) == ["metric", "value"]

    metrics = set(report["metric"])

    expected_metrics = {
        "n_prediction_rows",
        "n_predicted_seasons",
        "first_predicted_season",
        "last_predicted_season",
        "n_rows",
        "accuracy",
        "log_loss",
        "mean_home_prob",
        "mean_draw_prob",
        "mean_away_prob",
        "min_probability_sum",
        "max_probability_sum",
        "max_abs_probability_sum_error",
    }

    assert expected_metrics.issubset(metrics)

    report_lookup = dict(zip(report["metric"], report["value"]))

    assert report_lookup["n_prediction_rows"] == 6
    assert report_lookup["n_predicted_seasons"] == 2
    assert report_lookup["first_predicted_season"] == 2022
    assert report_lookup["last_predicted_season"] == 2023