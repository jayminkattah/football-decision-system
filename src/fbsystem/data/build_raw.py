from pathlib import Path

import pandas as pd

from fbsystem.data.external_inventory import read_external_match_file


SELECTED_LEAGUES: dict[str, list[str]] = {
    "england": ["E0"],
    "spain": ["SP1"],
    "germany": ["D1"],
    "italy": ["I1"],
    "france": ["F1"],
    "netherlands": ["N1"],
    "portugal": ["P1"],
    "belgium": ["B1"],
    "mexico": ["MEX"],
}


def select_inventory_files(
    inventory: pd.DataFrame,
    selected_leagues: dict[str, list[str]],
    min_season_start_year: int,
) -> pd.DataFrame:
    """
    Select external files for the MVP raw dataset.

    For normal season-folder files, only seasons from min_season_start_year onward
    are selected.

    For country-level files with season_code='all', the whole file is selected.
    Row-level season filtering will happen later during canonical cleaning.

    Parameters
    ----------
    inventory:
        External file inventory.
    selected_leagues:
        Mapping of country to allowed league codes.
    min_season_start_year:
        Minimum season start year to include.

    Returns
    -------
    pd.DataFrame
        Filtered inventory rows.

    Required input columns
    ----------------------
    country
    league_code
    season_code
    season_start_year
    load_error
    """
    selected_rows = []

    for country, league_codes in selected_leagues.items():
        country_mask = inventory["country"].str.lower() == country
        league_mask = inventory["league_code"].isin(league_codes)
        readable_mask = inventory["load_error"].fillna("") == ""

        season_folder_mask = (
            inventory["season_start_year"].notna()
            & (inventory["season_start_year"] >= min_season_start_year)
        )

        country_file_mask = inventory["season_code"].fillna("") == "all"

        selected_rows.append(
            inventory[
                country_mask
                & league_mask
                & readable_mask
                & (season_folder_mask | country_file_mask)
            ]
        )

    if not selected_rows:
        return pd.DataFrame(columns=inventory.columns)

    selected = pd.concat(selected_rows, ignore_index=True)

    return selected.sort_values(
        ["country", "source_layout", "season_start_year", "league_code"],
        na_position="last",
    ).reset_index(drop=True)


def build_raw_matches_from_inventory(selected_inventory: pd.DataFrame) -> pd.DataFrame:
    """
    Build one concatenated raw match dataset from selected external files.

    No cleaning, renaming, date parsing, odds processing, or result filtering is done.

    Added metadata columns
    ----------------------
    source_country:
        Country folder.
    source_season:
        Season folder code, or 'all' for whole-country files.
    source_season_start_year:
        Parsed season start year, if available.
    source_league_code:
        League file code.
    source_file:
        Original file path.
    source_layout:
        File layout found during external inventory.
    """
    frames: list[pd.DataFrame] = []

    for row in selected_inventory.itertuples(index=False):
        file_path = Path(row.file_path)

        df = read_external_match_file(file_path)

        df["source_country"] = row.country
        df["source_season"] = row.season_code
        df["source_season_start_year"] = row.season_start_year
        df["source_league_code"] = row.league_code
        df["source_file"] = row.file_path
        df["source_layout"] = row.source_layout

        frames.append(df)

    if not frames:
        raise ValueError("No selected files found for raw dataset build.")

    return pd.concat(frames, ignore_index=True, sort=False)


def save_raw_matches(raw_df: pd.DataFrame, output_path: str | Path) -> None:
    """
    Save the concatenated raw match dataset.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(path, index=False)