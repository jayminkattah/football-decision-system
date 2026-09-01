import pandas as pd
import pytest

from fbsystem.strategies.rules import build_strategy_opportunities, build_strategy_report


def _sample_matches() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": ["m1", "m2"],
            "match_date": pd.to_datetime(["2024-08-10", "2024-08-11"]),
            "season": ["2024-2025", "2024-2025"],
            "season_start_year": [2024, 2024],
            "source_country": ["ENG", "ENG"],
            "source_league_code": ["E0", "E0"],
            "league": ["Premier League", "Premier League"],
            "home_team": ["Alpha", "Gamma"],
            "away_team": ["Beta", "Delta"],
            "full_time_result": ["H", "A"],
            "home_odds": [2.0, 1.8],
            "draw_odds": [3.2, 3.5],
            "away_odds": [4.0, 5.0],
            "market_home_prob": [0.50, 0.55],
            "market_draw_prob": [0.30, 0.25],
            "market_away_prob": [0.20, 0.20],
            "model_home_prob": [0.53, 0.54],
            "model_draw_prob": [0.29, 0.28],
            "model_away_prob": [0.18, 0.18],
            "home_edge": [0.03, -0.01],
            "draw_edge": [-0.01, 0.03],
            "away_edge": [-0.02, -0.02],
        }
    )


def test_build_strategy_opportunities_creates_three_rows_per_match():
    opportunities = build_strategy_opportunities(_sample_matches(), min_edge=0.02)

    assert len(opportunities) == 6
    assert set(opportunities["outcome"]) == {"H", "D", "A"}
    assert opportunities["match_id"].nunique() == 2


def test_build_strategy_opportunities_applies_edge_threshold():
    opportunities = build_strategy_opportunities(_sample_matches(), min_edge=0.02)

    selected = opportunities.loc[opportunities["selected"], ["match_id", "outcome"]]

    assert selected.to_dict("records") == [
        {"match_id": "m1", "outcome": "H"},
        {"match_id": "m2", "outcome": "D"},
    ]


def test_build_strategy_report_counts_selected_opportunities():
    opportunities = build_strategy_opportunities(_sample_matches(), min_edge=0.02)
    report = build_strategy_report(opportunities, min_edge=0.02)

    assert report.loc[0, "n_matches"] == 2
    assert report.loc[0, "n_opportunities"] == 6
    assert report.loc[0, "n_selected"] == 2
    assert report.loc[0, "n_selected_home"] == 1
    assert report.loc[0, "n_selected_draw"] == 1
    assert report.loc[0, "n_selected_away"] == 0


def test_build_strategy_opportunities_rejects_negative_min_edge():
    with pytest.raises(ValueError, match="min_edge"):
        build_strategy_opportunities(_sample_matches(), min_edge=-0.01)


def test_build_strategy_opportunities_requires_expected_columns():
    matches = _sample_matches().drop(columns=["home_edge"])

    with pytest.raises(ValueError, match="home_edge"):
        build_strategy_opportunities(matches)
