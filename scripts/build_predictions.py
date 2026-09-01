from __future__ import annotations

from pathlib import Path

import pandas as pd

from fbsystem.features.edges import add_edges, build_edge_report
from fbsystem.models.walk_forward import build_predictions_report, generate_walk_forward_predictions


INPUT_PATH = Path("data/processed/matches_with_market_probs.parquet")
OUTPUT_PATH = Path("data/processed/matches_with_predictions.parquet")
REPORT_PATH = Path("outputs/evaluation/predictions_report.csv")
EDGE_REPORT_PATH = Path("outputs/evaluation/edge_report.csv")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}. "
            "Run `uv run python scripts/build_market_probabilities.py` first."
        )

    df = pd.read_parquet(INPUT_PATH)

    predictions = generate_walk_forward_predictions(df)

    if predictions.empty:
        raise ValueError(
            "No walk-forward predictions were generated. "
            "Check that the dataset contains at least two seasons and valid training data."
        )

    predictions_with_edges = add_edges(predictions)

    predictions_report = build_predictions_report(predictions_with_edges)
    edge_report = build_edge_report(predictions_with_edges)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    EDGE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    predictions_with_edges.to_parquet(OUTPUT_PATH, index=False)
    predictions_report.to_csv(REPORT_PATH, index=False)
    edge_report.to_csv(EDGE_REPORT_PATH, index=False)

    predicted_seasons = sorted(predictions_with_edges["prediction_test_season"].unique())

    print("Walk-forward prediction build complete.")
    print(f"Input rows: {len(df):,}")
    print(f"Prediction rows: {len(predictions_with_edges):,}")
    print(f"Output columns: {predictions_with_edges.shape[1]:,}")
    print(f"Predicted seasons: {predicted_seasons}")
    print(f"First predicted season: {min(predicted_seasons)}")
    print(f"Last predicted season: {max(predicted_seasons)}")
    print(f"Saved predictions to: {OUTPUT_PATH}")
    print(f"Saved prediction report to: {REPORT_PATH}")
    print(f"Saved edge report to: {EDGE_REPORT_PATH}")


if __name__ == "__main__":
    main()