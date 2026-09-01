from pathlib import Path

import pandas as pd

from fbsystem.data.inspect_raw import (
    build_column_schema_report,
    build_raw_dataset_summary,
    save_raw_schema_outputs,
)


def test_build_column_schema_report_has_expected_columns() -> None:
    df = pd.DataFrame(
        {
            "HomeTeam": ["Team A", "Team B", None],
            "FTHG": [2, 1, None],
            "B365H": [2.1, 1.8, 2.4],
        }
    )

    report = build_column_schema_report(df)

    expected_columns = [
        "column_name",
        "pandas_dtype",
        "non_null_count",
        "null_count",
        "null_percentage",
        "unique_count",
        "sample_values",
    ]

    assert list(report.columns) == expected_columns
    assert len(report) == 3


def test_build_raw_dataset_summary_returns_dataset_counts() -> None:
    df = pd.DataFrame(
        {
            "HomeTeam": ["Team A", "Team A"],
            "AwayTeam": ["Team B", "Team B"],
        }
    )

    summary = build_raw_dataset_summary(df)

    assert summary["row_count"] == 2
    assert summary["column_count"] == 2
    assert summary["columns"] == ["HomeTeam", "AwayTeam"]
    assert summary["duplicate_row_count"] == 1


def test_save_raw_schema_outputs_creates_files(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "Date": ["2023-08-12"],
            "HomeTeam": ["Team A"],
            "AwayTeam": ["Team B"],
        }
    )

    report = build_column_schema_report(df)

    save_raw_schema_outputs(
        df=df,
        schema_report=report,
        output_dir=tmp_path,
    )

    assert (tmp_path / "raw_schema_report.csv").exists()
    assert (tmp_path / "raw_schema_summary.txt").exists()