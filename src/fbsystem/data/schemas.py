import pandas as pd

try:
    import pandera.pandas as pa
except ImportError:  # pragma: no cover
    import pandera as pa

# Match dates are filtered to >= 2021-01-01.
# However, Jan-Jun 2021 matches belong to the 2020-2021 football season,
# so season_start_year can validly be 2020.


CANONICAL_MATCH_COLUMNS = [
    "match_id",
    "match_date",
    "season",
    "season_start_year",
    "source_country",
    "source_league_code",
    "league",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "full_time_result",
    "home_odds",
    "draw_odds",
    "away_odds",
    "odds_source",
    "source_season",
    "source_layout",
    "source_file",
]


CANONICAL_MATCH_SCHEMA = pa.DataFrameSchema(
    {
        "match_id": pa.Column(str, nullable=False, unique=True),
        "match_date": pa.Column("datetime64[ns]", nullable=False),
        "season": pa.Column(str, nullable=False),
        "season_start_year": pa.Column(int, nullable=False, checks=pa.Check.ge(2021)),
        "source_country": pa.Column(str, nullable=False),
        "source_league_code": pa.Column(str, nullable=False),
        "league": pa.Column(str, nullable=False),
        "home_team": pa.Column(str, nullable=False),
        "away_team": pa.Column(str, nullable=False),
        "home_goals": pa.Column(int, nullable=False, checks=pa.Check.ge(0)),
        "away_goals": pa.Column(int, nullable=False, checks=pa.Check.ge(0)),
        "full_time_result": pa.Column(
            str,
            nullable=False,
            checks=pa.Check.isin(["H", "D", "A"]),
        ),
        "home_odds": pa.Column(float, nullable=False, checks=pa.Check.gt(1.0)),
        "draw_odds": pa.Column(float, nullable=False, checks=pa.Check.gt(1.0)),
        "away_odds": pa.Column(float, nullable=False, checks=pa.Check.gt(1.0)),
        "odds_source": pa.Column(str, nullable=False),
        "source_season": pa.Column(str, nullable=False),
        "source_layout": pa.Column(str, nullable=False),
        "source_file": pa.Column(str, nullable=False),
    },
    strict=True,
    coerce=True,
)


def validate_canonical_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the canonical match dataset.

    Parameters
    ----------
    df:
        Canonical match-level DataFrame.

    Returns
    -------
    pd.DataFrame
        Validated canonical match-level DataFrame.
    """
    return CANONICAL_MATCH_SCHEMA.validate(df)