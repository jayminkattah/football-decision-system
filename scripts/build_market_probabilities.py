from __future__ import annotations

from pathlib import Path

import pandas as pd

from fbsystem.features.market_probs import (
    MARKET_OVERROUND_COLUMN,
    MARKET_PROB_COLUMNS,
    add_market_implied_probabilities,
    build_market_probability_report,
)


INPUT_PATH = Path("data/processed/matches_canonical.parquet")
OUTPUT_PATH = Path("data/processed/matches_with_market_probs.parquet")
REPORT_PATH = Path("outputs/evaluation/market_probability_report.csv")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}. "
            "Run `uv run python scripts/build_canonical_matches.py` first."
        )

    df = pd.read_parquet(INPUT_PATH)

    df_with_market_probs = add_market_implied_probabilities(df)
    report = build_market_probability_report(df_with_market_probs)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df_with_market_probs.to_parquet(OUTPUT_PATH, index=False)
    report.to_csv(REPORT_PATH, index=False)

    probability_sums = df_with_market_probs[MARKET_PROB_COLUMNS].sum(axis=1)
    max_probability_sum_error = float((probability_sums - 1.0).abs().max())

    print("Market probability build complete.")
    print(f"Input rows: {len(df):,}")
    print(f"Output rows: {len(df_with_market_probs):,}")
    print(f"Output columns: {df_with_market_probs.shape[1]:,}")
    print(f"Mean market overround: {df_with_market_probs[MARKET_OVERROUND_COLUMN].mean():.6f}")
    print(f"Min market overround: {df_with_market_probs[MARKET_OVERROUND_COLUMN].min():.6f}")
    print(f"Max market overround: {df_with_market_probs[MARKET_OVERROUND_COLUMN].max():.6f}")
    print(f"Max probability sum error: {max_probability_sum_error:.12f}")
    print(f"Saved market probability dataset to: {OUTPUT_PATH}")
    print(f"Saved market probability report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()