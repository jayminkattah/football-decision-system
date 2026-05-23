import pandas as pd

from fbsystem.data.clean_matches import (
    clean_canonical_matches,
    derive_result_from_score,
    derive_season_start_year,
)
from fbsystem.data.schemas import validate_canonical_matches


def test_derive_season_start_year() -> None:
    dates = pd.to_datetime(
        pd.Series(
            [
                "2021-05-01",
                "2021-07-01",
                "2022-01-15",
            ]
        )
    )

    result = derive_season_start_year(dates)

    assert result.tolist() == [2020, 2021, 2021]


def test_derive_result_from_score() -> None:
    home_goals = pd.Series([2, 1, 0])
    away_goals = pd.Series([1, 1, 3])

    result = derive_result_from_score(home_goals, away_goals)

    assert result.tolist() == ["H", "D", "A"]


def test_clean_canonical_matches_maps_standard_schema() -> None:
    raw_df = pd.DataFrame(
        {
            "Date": ["12/09/2021"],
            "Div": ["E0"],
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Chelsea"],
            "FTHG": [2],
            "FTAG": [1],
            "FTR": ["H"],
            "AvgCH": [2.10],
            "AvgCD": [3.40],
            "AvgCA": [3.20],
            "source_country": ["england"],
            "source_season": ["2021"],
            "source_season_start_year": [2020],
            "source_league_code": ["E0"],
            "source_file": ["data/external/england/2021/E0.csv"],
            "source_layout": ["country_season_file"],
        }
    )

    canonical_df, report_df = clean_canonical_matches(raw_df)

    assert len(canonical_df) == 1
    assert canonical_df.loc[0, "home_team"] == "Arsenal"
    assert canonical_df.loc[0, "away_team"] == "Chelsea"
    assert canonical_df.loc[0, "home_goals"] == 2
    assert canonical_df.loc[0, "away_goals"] == 1
    assert canonical_df.loc[0, "full_time_result"] == "H"
    assert canonical_df.loc[0, "home_odds"] == 2.10
    assert canonical_df.loc[0, "season"] == "2021-2022"
    assert "final_canonical_rows" in report_df["check"].tolist()

    validate_canonical_matches(canonical_df)


def test_clean_canonical_matches_maps_mexico_schema() -> None:
    raw_df = pd.DataFrame(
        {
            "Date": ["21/07/2021"],
            "Country": ["Mexico"],
            "League": ["Liga MX"],
            "Season": ["2021/2022"],
            "Home": ["Club America"],
            "Away": ["Atlas"],
            "HG": [1],
            "AG": [1],
            "Res": ["D"],
            "AvgCH": [1.90],
            "AvgCD": [3.30],
            "AvgCA": [4.20],
            "source_country": ["mexico"],
            "source_season": ["all"],
            "source_season_start_year": [pd.NA],
            "source_league_code": ["MEX"],
            "source_file": ["data/external/mexico/MEX.csv"],
            "source_layout": ["country_file"],
        }
    )

    canonical_df, _ = clean_canonical_matches(raw_df)

    assert len(canonical_df) == 1
    assert canonical_df.loc[0, "source_country"] == "mexico"
    assert canonical_df.loc[0, "league"] == "Liga MX"
    assert canonical_df.loc[0, "home_team"] == "Club America"
    assert canonical_df.loc[0, "away_team"] == "Atlas"
    assert canonical_df.loc[0, "full_time_result"] == "D"

    validate_canonical_matches(canonical_df)


def test_clean_canonical_matches_filters_pre_2021_rows() -> None:
    raw_df = pd.DataFrame(
        {
            "Date": ["21/067/2021", "21/07/2021"],
            "Home": ["Old Home", "New Home"],
            "Away": ["Old Away", "New Away"],
            "HG": [1, 2],
            "AG": [0, 1],
            "Res": ["H", "H"],
            "League": ["Liga MX", "Liga MX"],
            "AvgCH": [1.90, 2.10],
            "AvgCD": [3.30, 3.20],
            "AvgCA": [4.20, 3.80],
            "source_country": ["mexico", "mexico"],
            "source_season": ["all", "all"],
            "source_season_start_year": [pd.NA, pd.NA],
            "source_league_code": ["MEX", "MEX"],
            "source_file": ["data/external/mexico/MEX.csv", "data/external/mexico/MEX.csv"],
            "source_layout": ["country_file", "country_file"],
        }
    )

    canonical_df, _ = clean_canonical_matches(raw_df)

    assert len(canonical_df) == 1
    assert canonical_df.loc[0, "home_team"] == "New Home"


def test_clean_canonical_matches_filters_result_score_mismatch() -> None:
    raw_df = pd.DataFrame(
        {
            "Date": ["21/07/2020"],
            "Home": ["Club America"],
            "Away": ["Atlas"],
            "HG": [2],
            "AG": [0],
            "Res": ["A"],
            "League": ["Liga MX"],
            "AvgCH": [1.90],
            "AvgCD": [3.30],
            "AvgCA": [4.20],
            "source_country": ["mexico"],
            "source_season": ["all"],
            "source_season_start_year": [pd.NA],
            "source_league_code": ["MEX"],
            "source_file": ["data/external/mexico/MEX.csv"],
            "source_layout": ["country_file"],
        }
    )

    canonical_df, report_df = clean_canonical_matches(raw_df)

    mismatch_rows = report_df.loc[
        report_df["check"] == "result_score_mismatch",
        "row_count",
    ].iloc[0]

    assert len(canonical_df) == 0
    assert mismatch_rows == 1


def test_clean_canonical_matches_filters_invalid_odds() -> None:
    raw_df = pd.DataFrame(
        {
            "Date": ["21/07/2020"],
            "Home": ["Club America"],
            "Away": ["Atlas"],
            "HG": [1],
            "AG": [0],
            "Res": ["H"],
            "League": ["Liga MX"],
            "AvgCH": [1.00],
            "AvgCD": [3.30],
            "AvgCA": [4.20],
            "source_country": ["mexico"],
            "source_season": ["all"],
            "source_season_start_year": [pd.NA],
            "source_league_code": ["MEX"],
            "source_file": ["data/external/mexico/MEX.csv"],
            "source_layout": ["country_file"],
        }
    )

    canonical_df, report_df = clean_canonical_matches(raw_df)

    invalid_odds_rows = report_df.loc[
        report_df["check"] == "invalid_odds",
        "row_count",
    ].iloc[0]

    assert len(canonical_df) == 0
    assert invalid_odds_rows == 1