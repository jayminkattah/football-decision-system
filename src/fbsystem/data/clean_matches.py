from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from fbsystem.data.schemas import CANONICAL_MATCH_COLUMNS


MIN_MATCH_DATE = "2021-07-01"


def coalesce_columns(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    """
    Return the first non-null value across candidate columns.

    Missing candidate columns are ignored.
    """
    existing_columns = [column for column in candidates if column in df.columns]

    if not existing_columns:
        return pd.Series(pd.NA, index=df.index)

    output = df[existing_columns[0]].copy()

    for column in existing_columns[1:]:
        output = output.combine_first(df[column])

    return output


def clean_string_series(series: pd.Series) -> pd.Series:
    """
    Strip whitespace and collapse repeated internal spaces.
    """
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def derive_season_start_year(match_dates: pd.Series) -> pd.Series:
    """
    Derive football season start year from match date.

    July to December belongs to the current year.
    January to June belongs to the previous year.

    Example
    -------
    2021-05-10 -> 2020
    2021-09-10 -> 2021
    """
    years = match_dates.dt.year
    months = match_dates.dt.month

    season_start_year = np.where(months >= 7, years, years - 1)

    return pd.Series(season_start_year, index=match_dates.index).astype("Int64")


def build_season_label(season_start_year: pd.Series) -> pd.Series:
    """
    Build readable season labels.

    Example
    -------
    2020 -> 2020-2021
    """
    start = season_start_year.astype("Int64")
    end = start + 1

    return start.astype(str) + "-" + end.astype(str)


def derive_result_from_score(
    home_goals: pd.Series,
    away_goals: pd.Series,
) -> pd.Series:
    """
    Derive H/D/A result from full-time goals.
    """
    result = pd.Series(pd.NA, index=home_goals.index, dtype="object")

    result.loc[home_goals > away_goals] = "H"
    result.loc[home_goals == away_goals] = "D"
    result.loc[home_goals < away_goals] = "A"

    return result.astype("string")


def build_match_id(df: pd.DataFrame) -> pd.Series:
    """
    Build a stable match ID from canonical match identity fields.

    Returns an empty Series when df is empty so downstream assignment still works.
    """
    if df.empty:
        return pd.Series(index=df.index, dtype="string", name="match_id")

    key_columns = [
        "source_country",
        "source_league_code",
        "match_date",
        "home_team",
        "away_team",
    ]

    key_frame = df[key_columns].copy()
    key_frame["match_date"] = key_frame["match_date"].dt.strftime("%Y-%m-%d")

    key_strings = key_frame.astype(str).agg("|".join, axis=1)

    return key_strings.map(
        lambda value: hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    ).astype("string")
    

def map_raw_to_canonical(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map raw football match data into canonical columns before filtering.

    No rows are dropped in this function.
    """
    canonical = pd.DataFrame(index=raw_df.index)

    canonical["match_date"] = pd.to_datetime(
        coalesce_columns(raw_df, ["Date"]),
        errors="coerce",
        dayfirst=True,
    )

    canonical["source_country"] = clean_string_series(
        coalesce_columns(raw_df, ["source_country"])
    ).str.lower()

    canonical["source_league_code"] = clean_string_series(
        coalesce_columns(raw_df, ["source_league_code"])
    )

    canonical["league"] = clean_string_series(
        coalesce_columns(raw_df, ["Div", "League", "source_league_code"])
    )

    canonical["home_team"] = clean_string_series(
        coalesce_columns(raw_df, ["HomeTeam", "Home"])
    )

    canonical["away_team"] = clean_string_series(
        coalesce_columns(raw_df, ["AwayTeam", "Away"])
    )

    canonical["home_goals"] = pd.to_numeric(
        coalesce_columns(raw_df, ["FTHG", "HG"]),
        errors="coerce",
    )

    canonical["away_goals"] = pd.to_numeric(
        coalesce_columns(raw_df, ["FTAG", "AG"]),
        errors="coerce",
    )

    canonical["full_time_result"] = clean_string_series(
        coalesce_columns(raw_df, ["FTR", "Res"])
    ).str.upper()

    canonical["home_odds"] = pd.to_numeric(
        coalesce_columns(raw_df, ["AvgCH"]),
        errors="coerce",
    )

    canonical["draw_odds"] = pd.to_numeric(
        coalesce_columns(raw_df, ["AvgCD"]),
        errors="coerce",
    )

    canonical["away_odds"] = pd.to_numeric(
        coalesce_columns(raw_df, ["AvgCA"]),
        errors="coerce",
    )

    canonical["odds_source"] = "AvgC"

    canonical["source_season"] = clean_string_series(
        coalesce_columns(raw_df, ["source_season"])
    )

    canonical["source_layout"] = clean_string_series(
        coalesce_columns(raw_df, ["source_layout"])
    )

    canonical["source_file"] = clean_string_series(
        coalesce_columns(raw_df, ["source_file"])
    )

    return canonical


def build_cleaning_report(
    mapped_df: pd.DataFrame,
    min_match_date: str,
    final_df: pd.DataFrame,
    duplicate_match_id_rows: int,
) -> pd.DataFrame:
    """
    Build a row-count report for canonical cleaning.

    Reason counts are not mutually exclusive.
    A row can fail multiple checks.
    """
    min_date = pd.Timestamp(min_match_date)

    home_goals = mapped_df["home_goals"]
    away_goals = mapped_df["away_goals"]
    derived_result = derive_result_from_score(home_goals, away_goals)

    result_score_mismatch = (
        mapped_df["full_time_result"].notna()
        & derived_result.notna()
        & (mapped_df["full_time_result"] != derived_result)
    )

    records = [
        {
            "check": "raw_mapped_rows",
            "row_count": len(mapped_df),
            "description": "Rows after raw columns are mapped into canonical names.",
        },
        {
            "check": "invalid_or_missing_match_date",
            "row_count": int(mapped_df["match_date"].isna().sum()),
            "description": "Rows with missing or unparseable match_date.",
        },
        {
            "check": "before_min_match_date",
            "row_count": int((mapped_df["match_date"] < min_date).sum()),
            "description": f"Rows before {min_match_date}.",
        },
        {
            "check": "missing_team",
            "row_count": int(
                mapped_df["home_team"].isna().sum()
                + mapped_df["away_team"].isna().sum()
            ),
            "description": "Rows with missing home_team or away_team. Count is column-level, not row-level.",
        },
        {
            "check": "invalid_result",
            "row_count": int(
                (~mapped_df["full_time_result"].isin(["H", "D", "A"])).sum()
            ),
            "description": "Rows where full_time_result is not H, D, or A.",
        },
        {
            "check": "missing_goals",
            "row_count": int(
                mapped_df["home_goals"].isna().sum()
                + mapped_df["away_goals"].isna().sum()
            ),
            "description": "Rows with missing home_goals or away_goals. Count is column-level, not row-level.",
        },
        {
            "check": "result_score_mismatch",
            "row_count": int(result_score_mismatch.sum()),
            "description": "Rows where full_time_result disagrees with home_goals and away_goals.",
        },
        {
            "check": "invalid_odds",
            "row_count": int(
                (
                    mapped_df["home_odds"].isna()
                    | mapped_df["draw_odds"].isna()
                    | mapped_df["away_odds"].isna()
                    | (mapped_df["home_odds"] <= 1.0)
                    | (mapped_df["draw_odds"] <= 1.0)
                    | (mapped_df["away_odds"] <= 1.0)
                ).sum()
            ),
            "description": "Rows with missing odds or decimal odds <= 1.0.",
        },
        {
            "check": "duplicate_match_id_rows",
            "row_count": duplicate_match_id_rows,
            "description": "Rows with duplicated canonical match IDs after filtering.",
        },
        {
            "check": "final_canonical_rows",
            "row_count": len(final_df),
            "description": "Rows saved to canonical match dataset.",
        },
        {
            "check": "final_min_match_date",
            "row_count": 0,
            "description": str(final_df["match_date"].min()) if len(final_df) else "",
        },
        {
            "check": "final_max_match_date",
            "row_count": 0,
            "description": str(final_df["match_date"].max()) if len(final_df) else "",
        },
        {
            "check": "final_seasons",
            "row_count": int(final_df["season"].nunique()) if len(final_df) else 0,
            "description": ", ".join(sorted(final_df["season"].unique())) if len(final_df) else "",
        },
    ]

    return pd.DataFrame.from_records(records)


def clean_canonical_matches(
    raw_df: pd.DataFrame,
    min_match_date: str = MIN_MATCH_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean raw match data into the canonical match dataset.

    Parameters
    ----------
    raw_df:
        Raw concatenated match data from data/raw/matches_raw.csv.
    min_match_date:
        Earliest match_date to keep.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        canonical_df:
            Clean canonical match dataset.
        report_df:
            Cleaning report with row counts.
    """
    mapped = map_raw_to_canonical(raw_df)

    min_date = pd.Timestamp(min_match_date)

    derived_result = derive_result_from_score(
        mapped["home_goals"],
        mapped["away_goals"],
    )

    valid_mask = (
        mapped["match_date"].notna()
        & (mapped["match_date"] >= min_date)
        & mapped["source_country"].notna()
        & mapped["source_league_code"].notna()
        & mapped["league"].notna()
        & mapped["home_team"].notna()
        & mapped["away_team"].notna()
        & mapped["home_goals"].notna()
        & mapped["away_goals"].notna()
        & mapped["full_time_result"].isin(["H", "D", "A"])
        & derived_result.notna()
        & (mapped["full_time_result"] == derived_result)
        & mapped["home_odds"].notna()
        & mapped["draw_odds"].notna()
        & mapped["away_odds"].notna()
        & (mapped["home_odds"] > 1.0)
        & (mapped["draw_odds"] > 1.0)
        & (mapped["away_odds"] > 1.0)
        & mapped["source_season"].notna()
        & mapped["source_layout"].notna()
        & mapped["source_file"].notna()
    )

    canonical = mapped.loc[valid_mask].copy()

    canonical["home_goals"] = canonical["home_goals"].astype(int)
    canonical["away_goals"] = canonical["away_goals"].astype(int)

    canonical["season_start_year"] = derive_season_start_year(
        canonical["match_date"]
    ).astype(int)

    canonical["season"] = build_season_label(
        canonical["season_start_year"]
    )

    canonical["match_id"] = build_match_id(canonical)

    duplicate_match_id_rows = int(canonical.duplicated("match_id", keep=False).sum())

    canonical = (
        canonical.drop_duplicates("match_id", keep="first")
        .sort_values(
            [
                "match_date",
                "source_country",
                "source_league_code",
                "home_team",
                "away_team",
            ]
        )
        .reset_index(drop=True)
    )

    canonical = canonical[CANONICAL_MATCH_COLUMNS]

    report = build_cleaning_report(
        mapped_df=mapped,
        min_match_date=min_match_date,
        final_df=canonical,
        duplicate_match_id_rows=duplicate_match_id_rows,
    )

    return canonical, report


def save_canonical_matches(
    canonical_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Save canonical matches as Parquet.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canonical_df.to_parquet(path, index=False)


def save_canonical_report(
    report_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Save canonical cleaning report as CSV.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(path, index=False)