import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from fbsystem.models.baseline import (
    BASELINE_FEATURE_COLUMNS,
    MODEL_PROB_COLUMNS,
    build_baseline_model_report,
    build_baseline_pipeline,
    evaluate_baseline_probabilities,
    fit_baseline_model,
    predict_baseline_probabilities,
    validate_training_data,
)


def make_training_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_home_prob": [
                0.55,
                0.30,
                0.40,
                0.60,
                0.25,
                0.45,
                0.50,
                0.35,
                0.42,
            ],
            "market_draw_prob": [
                0.25,
                0.30,
                0.35,
                0.20,
                0.35,
                0.30,
                0.25,
                0.33,
                0.31,
            ],
            "market_away_prob": [
                0.20,
                0.40,
                0.25,
                0.20,
                0.40,
                0.25,
                0.25,
                0.32,
                0.27,
            ],
            "season_start_year": [
                2021,
                2021,
                2021,
                2022,
                2022,
                2022,
                2023,
                2023,
                2023,
            ],
            "source_country": [
                "england",
                "england",
                "spain",
                "spain",
                "germany",
                "germany",
                "italy",
                "italy",
                "france",
            ],
            "source_league_code": [
                "E0",
                "E0",
                "SP1",
                "SP1",
                "D1",
                "D1",
                "I1",
                "I1",
                "F1",
            ],
            "full_time_result": [
                "H",
                "A",
                "D",
                "H",
                "A",
                "D",
                "H",
                "A",
                "D",
            ],
        }
    )


def test_build_baseline_pipeline_returns_pipeline():
    model = build_baseline_pipeline()

    assert isinstance(model, Pipeline)
    assert "preprocessor" in model.named_steps
    assert "classifier" in model.named_steps


def test_validate_training_data_accepts_valid_training_data():
    df = make_training_df()

    validate_training_data(df)


def test_validate_training_data_raises_for_missing_required_columns():
    df = make_training_df().drop(columns=["market_home_prob"])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_training_data(df)


def test_validate_training_data_raises_for_empty_dataframe():
    df = make_training_df().iloc[0:0]

    with pytest.raises(ValueError, match="Training data is empty"):
        validate_training_data(df)


def test_validate_training_data_raises_for_invalid_target_values():
    df = make_training_df()
    df.loc[0, "full_time_result"] = "X"

    with pytest.raises(ValueError, match="invalid target values"):
        validate_training_data(df)


def test_validate_training_data_raises_when_a_class_is_missing():
    df = make_training_df()
    df = df[df["full_time_result"] != "D"]

    with pytest.raises(ValueError, match="Missing classes"):
        validate_training_data(df)


def test_fit_baseline_model_returns_fitted_pipeline():
    df = make_training_df()

    model = fit_baseline_model(df)

    assert isinstance(model, Pipeline)
    assert hasattr(model.named_steps["classifier"], "classes_")


def test_predict_baseline_probabilities_returns_expected_columns():
    df = make_training_df()
    model = fit_baseline_model(df)

    probabilities = predict_baseline_probabilities(model, df)

    assert list(probabilities.columns) == MODEL_PROB_COLUMNS
    assert len(probabilities) == len(df)


def test_predict_baseline_probabilities_sum_to_one():
    df = make_training_df()
    model = fit_baseline_model(df)

    probabilities = predict_baseline_probabilities(model, df)

    probability_sums = probabilities[MODEL_PROB_COLUMNS].sum(axis=1)

    assert probability_sums.min() == pytest.approx(1.0)
    assert probability_sums.max() == pytest.approx(1.0)


def test_predict_baseline_probabilities_preserves_input_index():
    df = make_training_df()
    df.index = [10, 11, 12, 13, 14, 15, 16, 17, 18]

    model = fit_baseline_model(df)
    probabilities = predict_baseline_probabilities(model, df)

    assert probabilities.index.tolist() == df.index.tolist()


def test_predict_baseline_probabilities_raises_for_missing_feature_columns():
    df = make_training_df()
    model = fit_baseline_model(df)

    prediction_df = df.drop(columns=["source_country"])

    with pytest.raises(ValueError, match="Missing required columns"):
        predict_baseline_probabilities(model, prediction_df)


def test_evaluate_baseline_probabilities_returns_expected_metrics():
    df = make_training_df()
    model = fit_baseline_model(df)
    probabilities = predict_baseline_probabilities(model, df)

    metrics = evaluate_baseline_probabilities(
        y_true=df["full_time_result"],
        probability_df=probabilities,
    )

    expected_metrics = {
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

    assert expected_metrics.issubset(metrics.keys())
    assert metrics["n_rows"] == len(df)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["log_loss"] > 0.0
    assert metrics["max_abs_probability_sum_error"] == pytest.approx(0.0)


def test_build_baseline_model_report_returns_dataframe():
    metrics = {
        "n_rows": 10,
        "accuracy": 0.5,
        "log_loss": 1.1,
    }

    report = build_baseline_model_report(metrics)

    assert list(report.columns) == ["metric", "value"]
    assert len(report) == 3


def test_baseline_feature_columns_do_not_include_post_match_result_columns():
    forbidden_columns = {
        "home_goals",
        "away_goals",
        "full_time_result",
        "home_edge",
        "draw_edge",
        "away_edge",
    }

    assert forbidden_columns.isdisjoint(BASELINE_FEATURE_COLUMNS)