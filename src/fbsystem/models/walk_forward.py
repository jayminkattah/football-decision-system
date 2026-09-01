from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from fbsystem.models.baseline import (
    BASELINE_FEATURE_COLUMNS,
    MODEL_PROB_COLUMNS,
    TARGET_COLUMN,
    build_baseline_model_report,
    evaluate_baseline_probabilities,
    fit_baseline_model,
    predict_baseline_probabilities,
)


SEASON_COLUMN = "season_start_year"

WALK_FORWARD_METADATA_COLUMNS = [
    "prediction_train_start_season",
    "prediction_train_end_season",
    "prediction_test_season",
    "prediction_n_train_rows",
    "prediction_n_test_rows",
]


def validate_required_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def get_sorted_seasons(df: pd.DataFrame, season_column: str = SEASON_COLUMN) -> list[int]:
    """
    Return sorted non-null seasons.
    """
    validate_required_columns(df, [season_column])

    seasons = (
        df[season_column]
        .dropna()
        .astype(int)
        .sort_values()
        .unique()
        .tolist()
    )

    return seasons


def generate_walk_forward_predictions(
    df: pd.DataFrame,
    season_column: str = SEASON_COLUMN,
    min_train_seasons: int = 1,
    max_iter: int = 1_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate walk-forward model predictions.

    Rule
    ----
    For each test season:

    - train on all seasons strictly before the test season
    - predict on the test season
    - skip seasons that do not have enough prior training seasons

    This prevents future leakage.

    Output
    ------
    Original test rows plus:

    - model_home_prob
    - model_draw_prob
    - model_away_prob
    - prediction_train_start_season
    - prediction_train_end_season
    - prediction_test_season
    - prediction_n_train_rows
    - prediction_n_test_rows
    """
    required_columns = [
        season_column,
        TARGET_COLUMN,
        *BASELINE_FEATURE_COLUMNS,
    ]
    validate_required_columns(df, required_columns)

    if df.empty:
        output_columns = [
            *df.columns.tolist(),
            *MODEL_PROB_COLUMNS,
            *WALK_FORWARD_METADATA_COLUMNS,
        ]
        return pd.DataFrame(columns=output_columns)

    seasons = get_sorted_seasons(df, season_column=season_column)

    prediction_frames: list[pd.DataFrame] = []

    for test_season in seasons:
        train_seasons = [season for season in seasons if season < test_season]

        if len(train_seasons) < min_train_seasons:
            continue

        train_df = df[df[season_column] < test_season].copy()
        test_df = df[df[season_column] == test_season].copy()

        if train_df.empty or test_df.empty:
            continue

        model = fit_baseline_model(
            train_df,
            max_iter=max_iter,
            random_state=random_state,
        )

        probabilities = predict_baseline_probabilities(model, test_df)

        fold_predictions = test_df.copy()

        for column in MODEL_PROB_COLUMNS:
            fold_predictions[column] = probabilities[column]

        fold_predictions["prediction_train_start_season"] = int(train_df[season_column].min())
        fold_predictions["prediction_train_end_season"] = int(train_df[season_column].max())
        fold_predictions["prediction_test_season"] = int(test_season)
        fold_predictions["prediction_n_train_rows"] = int(len(train_df))
        fold_predictions["prediction_n_test_rows"] = int(len(test_df))

        prediction_frames.append(fold_predictions)

    if not prediction_frames:
        output_columns = [
            *df.columns.tolist(),
            *MODEL_PROB_COLUMNS,
            *WALK_FORWARD_METADATA_COLUMNS,
        ]
        return pd.DataFrame(columns=output_columns)

    predictions = pd.concat(prediction_frames, axis=0).sort_index()

    probability_sums = predictions[MODEL_PROB_COLUMNS].sum(axis=1)

    if not np.allclose(probability_sums, 1.0, atol=1e-6, rtol=0.0):
        max_error = float((probability_sums - 1.0).abs().max())
        raise ValueError(
            "Walk-forward model probabilities do not sum to 1.0 within tolerance. "
            f"Max absolute error: {max_error}"
        )

    return predictions


def build_predictions_report(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Build a simple report for walk-forward predictions.
    """
    required_columns = [
        TARGET_COLUMN,
        *MODEL_PROB_COLUMNS,
        *WALK_FORWARD_METADATA_COLUMNS,
    ]
    validate_required_columns(predictions, required_columns)

    if predictions.empty:
        rows = [
            ("n_prediction_rows", 0),
            ("n_predicted_seasons", 0),
            ("first_predicted_season", np.nan),
            ("last_predicted_season", np.nan),
            ("accuracy", np.nan),
            ("log_loss", np.nan),
            ("mean_home_prob", np.nan),
            ("mean_draw_prob", np.nan),
            ("mean_away_prob", np.nan),
            ("min_probability_sum", np.nan),
            ("max_probability_sum", np.nan),
            ("max_abs_probability_sum_error", np.nan),
        ]
        return pd.DataFrame(rows, columns=["metric", "value"])

    metrics = evaluate_baseline_probabilities(
        y_true=predictions[TARGET_COLUMN],
        probability_df=predictions[MODEL_PROB_COLUMNS],
    )

    report = build_baseline_model_report(metrics)

    extra_rows = pd.DataFrame(
        [
            {
                "metric": "n_prediction_rows",
                "value": int(len(predictions)),
            },
            {
                "metric": "n_predicted_seasons",
                "value": int(predictions["prediction_test_season"].nunique()),
            },
            {
                "metric": "first_predicted_season",
                "value": int(predictions["prediction_test_season"].min()),
            },
            {
                "metric": "last_predicted_season",
                "value": int(predictions["prediction_test_season"].max()),
            },
            {
                "metric": "first_train_start_season",
                "value": int(predictions["prediction_train_start_season"].min()),
            },
            {
                "metric": "last_train_end_season",
                "value": int(predictions["prediction_train_end_season"].max()),
            },
        ]
    )

    report = pd.concat([extra_rows, report], axis=0, ignore_index=True)

    return report