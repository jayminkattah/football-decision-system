from pathlib import Path

import pandas as pd


def load_raw_matches(file_path: str | Path) -> pd.DataFrame:
    """
    Load raw football match data from a CSV file.

    This function performs no cleaning and no schema enforcement.
    It only loads the raw data exactly as provided.

    Parameters
    ----------
    file_path:
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Raw match-level dataset.

    Expected input
    --------------
    A CSV file containing football match data and betting odds.

    Output columns
    --------------
    Same as the raw CSV file.
    No columns are renamed in this function.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a CSV file, got: {path.suffix}")

    df = pd.read_csv(path, low_memory=False)

    if df.empty:
        raise ValueError(f"Raw data file is empty: {path}")

    return df