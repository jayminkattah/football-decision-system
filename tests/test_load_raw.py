from pathlib import Path

import pandas as pd
import pytest

from fbsystem.data.load_raw import load_raw_matches


def test_load_raw_matches_loads_csv(tmp_path: Path) -> None:
    file_path = tmp_path / "matches.csv"

    input_df = pd.DataFrame(
        {
            "Date": ["2023-08-12"],
            "HomeTeam": ["Team A"],
            "AwayTeam": ["Team B"],
            "FTHG": [2],
            "FTAG": [1],
            "FTR": ["H"],
            "B365H": [2.1],
            "B365D": [3.4],
            "B365A": [3.2],
        }
    )

    input_df.to_csv(file_path, index=False)

    loaded_df = load_raw_matches(file_path)

    assert loaded_df.shape == input_df.shape
    assert list(loaded_df.columns) == list(input_df.columns)


def test_load_raw_matches_raises_for_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_raw_matches("missing_file.csv")


def test_load_raw_matches_raises_for_non_csv_file(tmp_path: Path) -> None:
    file_path = tmp_path / "matches.txt"
    file_path.write_text("not,a,csv")

    with pytest.raises(ValueError):
        load_raw_matches(file_path)


def test_load_raw_matches_raises_for_empty_csv(tmp_path: Path) -> None:
    file_path = tmp_path / "empty.csv"
    file_path.write_text("")

    with pytest.raises(ValueError):
        load_raw_matches(file_path)