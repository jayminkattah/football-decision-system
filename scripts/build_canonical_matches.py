from pathlib import Path

import pandas as pd

from fbsystem.data.clean_matches import (
    clean_canonical_matches,
    save_canonical_matches,
    save_canonical_report,
)
from fbsystem.data.load_raw import load_raw_matches
from fbsystem.data.schemas import validate_canonical_matches


RAW_DATA_PATH = Path("data/raw/matches_raw.csv")
CANONICAL_OUTPUT_PATH = Path("data/processed/matches_canonical.parquet")
REPORT_OUTPUT_PATH = Path("outputs/evaluation/canonical_data_report.csv")


def main() -> None:
    raw_df = load_raw_matches(RAW_DATA_PATH)

    canonical_df, report_df = clean_canonical_matches(raw_df)

    canonical_df = validate_canonical_matches(canonical_df)

    save_canonical_matches(
        canonical_df=canonical_df,
        output_path=CANONICAL_OUTPUT_PATH,
    )

    save_canonical_report(
        report_df=report_df,
        output_path=REPORT_OUTPUT_PATH,
    )

    print("Canonical match dataset build complete.")
    print(f"Raw rows: {len(raw_df):,}")
    print(f"Canonical rows: {len(canonical_df):,}")
    print(f"Columns: {len(canonical_df.columns):,}")
    print(f"Min date: {canonical_df['match_date'].min()}")
    print(f"Max date: {canonical_df['match_date'].max()}")
    print(f"Seasons: {', '.join(sorted(canonical_df['season'].unique()))}")
    print(f"Saved canonical dataset to: {CANONICAL_OUTPUT_PATH}")
    print(f"Saved cleaning report to: {REPORT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()