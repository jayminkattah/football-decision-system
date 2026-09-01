from pathlib import Path

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xls", ".xlsx"}


def season_code_to_start_year(season_code: str) -> int:
    """
    Convert football-data style season folder names to start years.

    Examples
    --------
    1920 -> 2019
    2021 -> 2020
    2122 -> 2021
    9900 -> 1999
    0001 -> 2000
    """
    if not season_code.isdigit() or len(season_code) != 4:
        raise ValueError(f"Invalid season code: {season_code}")

    first_two_digits = int(season_code[:2])

    if first_two_digits >= 90:
        return 1900 + first_two_digits

    return 2000 + first_two_digits


def read_external_match_file(file_path: str | Path) -> pd.DataFrame:
    """
    Read one external football data file.

    Supports CSV, XLS, and XLSX files.
    No cleaning is performed.

    Note
    ----
    If XLS/XLSX files fail to load, convert them to CSV manually.
    We are not adding extra Excel dependencies for the MVP.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(path)

    raise ValueError(f"Unsupported file type: {path}")


def _build_file_record(
    *,
    country: str,
    season_code: str | None,
    season_start_year: int | None,
    file_path: Path,
    source_layout: str,
) -> dict[str, object]:
    suffix = file_path.suffix.lower()

    try:
        df = read_external_match_file(file_path)
        row_count = len(df)
        column_count = len(df.columns)
        columns = list(df.columns)
        load_error = ""
    except Exception as exc:
        row_count = None
        column_count = None
        columns = []
        load_error = str(exc)

    return {
        "country": country,
        "season_code": season_code,
        "season_start_year": season_start_year,
        "file_name": file_path.name,
        "league_code": file_path.stem,
        "file_path": str(file_path),
        "file_extension": suffix,
        "source_layout": source_layout,
        "row_count": row_count,
        "column_count": column_count,
        "columns": columns,
        "load_error": load_error,
    }


def build_external_file_inventory(external_dir: str | Path) -> pd.DataFrame:
    """
    Build an inventory of files under data/external.

    Supported folder structures
    ---------------------------
    Standard season layout:

        data/external/{country}/{season}/{file}

    Whole-country file layout:

        data/external/{country}/{file}

    Returns
    -------
    pd.DataFrame

    Output columns
    --------------
    country:
        Country folder name.
    season_code:
        Season folder name, e.g. 1920. Uses 'all' for country-level files.
    season_start_year:
        Parsed start year, e.g. 2019. Null for country-level files.
    file_name:
        File name, e.g. B1.csv or MEX.csv.
    league_code:
        File stem, e.g. B1 or MEX.
    file_path:
        Full file path.
    file_extension:
        File extension.
    source_layout:
        Either 'country_season_file' or 'country_file'.
    row_count:
        Number of rows in the file.
    column_count:
        Number of columns in the file.
    columns:
        Original columns found in the file.
    load_error:
        Error message if the file could not be read.
    """
    root = Path(external_dir)

    if not root.exists():
        raise FileNotFoundError(f"External data directory not found: {root}")

    records: list[dict[str, object]] = []

    for country_dir in sorted(root.iterdir()):
        if not country_dir.is_dir():
            continue

        country = country_dir.name.lower()

        for child_path in sorted(country_dir.iterdir()):
            if child_path.is_file():
                suffix = child_path.suffix.lower()

                if suffix not in SUPPORTED_EXTENSIONS:
                    continue

                records.append(
                    _build_file_record(
                        country=country,
                        season_code="all",
                        season_start_year=None,
                        file_path=child_path,
                        source_layout="country_file",
                    )
                )

                continue

            if not child_path.is_dir():
                continue

            season_dir = child_path
            season_code = season_dir.name

            try:
                season_start_year = season_code_to_start_year(season_code)
            except ValueError:
                season_start_year = None

            for file_path in sorted(season_dir.iterdir()):
                if not file_path.is_file():
                    continue

                suffix = file_path.suffix.lower()

                if suffix not in SUPPORTED_EXTENSIONS:
                    continue

                records.append(
                    _build_file_record(
                        country=country,
                        season_code=season_code,
                        season_start_year=season_start_year,
                        file_path=file_path,
                        source_layout="country_season_file",
                    )
                )

    return pd.DataFrame.from_records(records)