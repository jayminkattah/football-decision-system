from __future__ import annotations

import inspect
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "full_time_result"

OUTCOME_CLASSES = ["H", "D", "A"]

NUMERIC_FEATURE_COLUMNS = [
    "market_home_prob",
    "market_draw_prob",
    "market_away_prob",
    "season_start_year",
]

CATEGORICAL_FEATURE_COLUMNS = [
    "source_country",
    "source_league_code",
]

BASELINE_FEATURE_COLUMNS = [
    *NUMERIC_FEATURE_COLUMNS,
    *CATEGORICAL_FEATURE_COLUMNS,
]

MODEL_PROB_COLUMNS = [
    "model_home_prob",
    "model_draw_prob",
    "model_away_prob",
]

CLASS_TO_PROB_COLUMN = {
    "H": "model_home_prob",
    "D": "model_draw_prob",
    "A": "model_away_prob",
}


def validate_required_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """
    Validate that all required columns exist in the DataFrame.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def validate_training_data(df: pd.DataFrame) -> None:
    """
    Validate training data for the baseline probability model.

    Required input columns
    ----------------------
    market_home_prob
    market_draw_prob
    market_away_prob
    season_start_year
    source_country
    source_league_code
    full_time_result

    Notes
    -----
    The baseline model intentionally uses only pre-match information.
    It does not use goals, results-derived features, or post-match stats.
    """
    required_columns = [*BASELINE_FEATURE_COLUMNS, TARGET_COLUMN]
    validate_required_columns(df, required_columns)

    if df.empty:
        raise ValueError("Training data is empty.")

    missing_target_count = int(df[TARGET_COLUMN].isna().sum())
    if missing_target_count > 0:
        raise ValueError(f"Found {missing_target_count} rows with missing target values.")

    invalid_target_mask = ~df[TARGET_COLUMN].isin(OUTCOME_CLASSES)
    if invalid_target_mask.any():
        invalid_values = sorted(df.loc[invalid_target_mask, TARGET_COLUMN].dropna().unique())
        raise ValueError(f"Found invalid target values: {invalid_values}")

    observed_classes = sorted(df[TARGET_COLUMN].unique())
    missing_classes = sorted(set(OUTCOME_CLASSES) - set(observed_classes))

    if missing_classes:
        raise ValueError(
            "Training data must contain all outcome classes "
            f"{OUTCOME_CLASSES}. Missing classes: {missing_classes}"
        )


def _make_one_hot_encoder() -> OneHotEncoder:
    """
    Create a OneHotEncoder compatible with multiple scikit-learn versions.

    scikit-learn renamed `sparse` to `sparse_output`.
    Because libraries enjoy tiny breaking changes. Naturally.
    """
    params = inspect.signature(OneHotEncoder).parameters

    if "sparse_output" in params:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_baseline_pipeline(
    max_iter: int = 1_000,
    random_state: int = 42,
) -> Pipeline:
    """
    Build a simple multinomial-style baseline classifier.

    Numeric features:
    - market_home_prob
    - market_draw_prob
    - market_away_prob
    - season_start_year

    Categorical features:
    - source_country
    - source_league_code

    Model:
    - LogisticRegression

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted scikit-learn pipeline.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", _make_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURE_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURE_COLUMNS),
        ],
        remainder="drop",
    )

    classifier = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def fit_baseline_model(
    df: pd.DataFrame,
    max_iter: int = 1_000,
    random_state: int = 42,
) -> Pipeline:
    """
    Fit the baseline probability model.

    Parameters
    ----------
    df:
        Training DataFrame containing baseline features and target.
    max_iter:
        Maximum number of iterations for LogisticRegression.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Fitted model pipeline.
    """
    validate_training_data(df)

    model = build_baseline_pipeline(
        max_iter=max_iter,
        random_state=random_state,
    )

    X = df[BASELINE_FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    model.fit(X, y)

    return model


def predict_baseline_probabilities(
    model: Pipeline,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Predict H/D/A probabilities using a fitted baseline model.

    Parameters
    ----------
    model:
        Fitted baseline model.
    df:
        DataFrame containing baseline feature columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with:

        - model_home_prob
        - model_draw_prob
        - model_away_prob
    """
    validate_required_columns(df, BASELINE_FEATURE_COLUMNS)

    X = df[BASELINE_FEATURE_COLUMNS]

    probabilities = model.predict_proba(X)
    classifier = model.named_steps["classifier"]
    fitted_classes = list(classifier.classes_)

    result = pd.DataFrame(index=df.index)

    for outcome_class, prob_column in CLASS_TO_PROB_COLUMN.items():
        if outcome_class in fitted_classes:
            class_index = fitted_classes.index(outcome_class)
            result[prob_column] = probabilities[:, class_index]
        else:
            result[prob_column] = 0.0

    probability_sums = result[MODEL_PROB_COLUMNS].sum(axis=1)

    if len(result) > 0 and not np.allclose(probability_sums, 1.0, atol=1e-6, rtol=0.0):
        max_error = float((probability_sums - 1.0).abs().max())
        raise ValueError(
            "Model probabilities do not sum to 1.0 within tolerance. "
            f"Max absolute error: {max_error}"
        )

    return result


def evaluate_baseline_probabilities(
    y_true: pd.Series,
    probability_df: pd.DataFrame,
) -> dict[str, float]:
    """
    Evaluate predicted probabilities.

    Metrics
    -------
    accuracy:
        Accuracy from the highest-probability predicted class.

    log_loss:
        Multiclass log loss.

    mean_home_prob:
        Average predicted home-win probability.

    mean_draw_prob:
        Average predicted draw probability.

    mean_away_prob:
        Average predicted away-win probability.
    """
    validate_required_columns(probability_df, MODEL_PROB_COLUMNS)

    if y_true.empty:
        raise ValueError("y_true is empty.")

    invalid_target_mask = ~y_true.isin(OUTCOME_CLASSES)
    if invalid_target_mask.any():
        invalid_values = sorted(y_true.loc[invalid_target_mask].dropna().unique())
        raise ValueError(f"Found invalid target values: {invalid_values}")

    probability_sums = probability_df[MODEL_PROB_COLUMNS].sum(axis=1)

    if not np.allclose(probability_sums, 1.0, atol=1e-6, rtol=0.0):
        max_error = float((probability_sums - 1.0).abs().max())
        raise ValueError(
            "Model probabilities do not sum to 1.0 within tolerance. "
            f"Max absolute error: {max_error}"
        )

    predicted_class_indices = probability_df[MODEL_PROB_COLUMNS].to_numpy().argmax(axis=1)
    predicted_classes = pd.Series(
        [OUTCOME_CLASSES[index] for index in predicted_class_indices],
        index=probability_df.index,
    )

    metrics = {
        "n_rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, predicted_classes)),
        "log_loss": float(
            log_loss(
                y_true,
                probability_df[MODEL_PROB_COLUMNS],
                labels=OUTCOME_CLASSES,
            )
        ),
        "mean_home_prob": float(probability_df["model_home_prob"].mean()),
        "mean_draw_prob": float(probability_df["model_draw_prob"].mean()),
        "mean_away_prob": float(probability_df["model_away_prob"].mean()),
        "min_probability_sum": float(probability_sums.min()),
        "max_probability_sum": float(probability_sums.max()),
        "max_abs_probability_sum_error": float((probability_sums - 1.0).abs().max()),
    }

    return metrics


def build_baseline_model_report(
    metrics: dict[str, float],
) -> pd.DataFrame:
    """
    Convert baseline model metrics into a report DataFrame.
    """
    return pd.DataFrame(
        [{"metric": metric, "value": value} for metric, value in metrics.items()]
    )