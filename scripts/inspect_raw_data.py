from pathlib import Path

from fbsystem.data.inspect_raw import (
    build_column_schema_report,
    save_raw_schema_outputs,
)
from fbsystem.data.load_raw import load_raw_matches


RAW_DATA_PATH = Path("data/raw/matches_raw.csv")
OUTPUT_DIR = Path("outputs/evaluation")


def main() -> None:
    raw_df = load_raw_matches(RAW_DATA_PATH)

    schema_report = build_column_schema_report(raw_df)

    save_raw_schema_outputs(
        df=raw_df,
        schema_report=schema_report,
        output_dir=OUTPUT_DIR,
    )

    print("Raw data inspection complete.")
    print(f"Rows: {len(raw_df)}")
    print(f"Columns: {len(raw_df.columns)}")
    print(f"Saved schema report to: {OUTPUT_DIR / 'raw_schema_report.csv'}")
    print(f"Saved summary to: {OUTPUT_DIR / 'raw_schema_summary.txt'}")


if __name__ == "__main__":
    main()