from pathlib import Path

import pandas as pd


def build_column_schema_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a column-level schema report for a raw DataFrame.

    Parameters
    ----------
    df:
        Raw match-level dataset.

    Returns
    -------
    pd.DataFrame
        One row per input column.

    Output columns
    --------------
    column_name:
        Original raw column name.
    pandas_dtype:
        Data type inferred by pandas.
    non_null_count:
        Number of non-missing values.
    null_count:
        Number of missing values.
    null_percentage:
        Percentage of rows with missing values.
    unique_count:
        Number of unique non-null values.
    sample_values:
        Up to five sample non-null values from the column.
    """
    row_count = len(df)

    records: list[dict[str, object]] = []

    for column in df.columns:
        series = df[column]
        non_null_series = series.dropna()

        sample_values = (
            non_null_series.astype(str)
            .drop_duplicates()
            .head(5)
            .tolist()
        )

        null_count = int(series.isna().sum())

        records.append(
            {
                "column_name": column,
                "pandas_dtype": str(series.dtype),
                "non_null_count": int(series.notna().sum()),
                "null_count": null_count,
                "null_percentage": round((null_count / row_count) * 100, 2)
                if row_count > 0
                else 0.0,
                "unique_count": int(non_null_series.nunique()),
                "sample_values": sample_values,
            }
        )

    return pd.DataFrame.from_records(records)


def build_raw_dataset_summary(df: pd.DataFrame) -> dict[str, object]:
    """
    Build a dataset-level summary for the raw match data.

    Parameters
    ----------
    df:
        Raw match-level dataset.

    Returns
    -------
    dict[str, object]
        Dataset-level summary.

    Output keys
    -----------
    row_count:
        Number of rows in the raw dataset.
    column_count:
        Number of columns in the raw dataset.
    columns:
        List of original raw column names.
    duplicate_row_count:
        Number of fully duplicated rows.
    """
    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": list(df.columns),
        "duplicate_row_count": int(df.duplicated().sum()),
    }


def save_raw_schema_outputs(
    df: pd.DataFrame,
    schema_report: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """
    Save raw schema inspection outputs.

    Parameters
    ----------
    df:
        Raw match-level dataset.
    schema_report:
        Column-level schema report.
    output_dir:
        Directory where outputs should be written.

    Files created
    -------------
    raw_schema_report.csv:
        Column-level schema report.
    raw_schema_summary.txt:
        Dataset-level summary and first five rows.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    schema_report_path = output_path / "raw_schema_report.csv"
    summary_path = output_path / "raw_schema_summary.txt"

    schema_report.to_csv(schema_report_path, index=False)

    summary = build_raw_dataset_summary(df)

    with summary_path.open("w", encoding="utf-8") as file:
        file.write("Raw Schema Summary\n")
        file.write("==================\n\n")

        file.write(f"Rows: {summary['row_count']}\n")
        file.write(f"Columns: {summary['column_count']}\n")
        file.write(f"Fully duplicated rows: {summary['duplicate_row_count']}\n\n")

        file.write("Column names:\n")
        for column in summary["columns"]:
            file.write(f"- {column}\n")

        file.write("\nFirst five rows:\n")
        file.write(df.head().to_string(index=False))
        file.write("\n")