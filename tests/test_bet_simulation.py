import pandas as pd
import pytest

from fbsystem.backtest.simulate import build_bet_simulation_report, simulate_flat_bets


def _sample_opportunities() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": ["m1", "m1", "m2"],
            "match_date": pd.to_datetime(["2024-08-10", "2024-08-10", "2024-08-11"]),
            "season": ["2024-2025", "2024-2025", "2024-2025"],
            "season_start_year": [2024, 2024, 2024],
            "source_country": ["ENG", "ENG", "ENG"],
            "source_league_code": ["E0", "E0", "E0"],
            "league": ["Premier League", "Premier League", "Premier League"],
            "home_team": ["Alpha", "Alpha", "Gamma"],
            "away_team": ["Beta", "Beta", "Delta"],
            "full_time_result": ["H", "H", "A"],
            "outcome": ["H", "D", "D"],
            "outcome_label": ["Home win", "Draw", "Draw"],
            "odds": [2.5, 3.2, 3.5],
            "market_prob": [0.40, 0.30, 0.25],
            "model_prob": [0.44, 0.32, 0.28],
            "edge": [0.04, 0.02, 0.03],
            "selected": [True, False, True],
        }
    )


def test_simulate_flat_bets_only_uses_selected_rows():
    bets = simulate_flat_bets(_sample_opportunities(), stake=1.0)

    assert len(bets) == 2
    assert bets["match_id"].tolist() == ["m1", "m2"]
    assert bets["stake"].tolist() == [1.0, 1.0]


def test_simulate_flat_bets_calculates_win_and_loss_profit():
    bets = simulate_flat_bets(_sample_opportunities(), stake=1.0)

    assert bets["is_win"].tolist() == [True, False]
    assert bets["return"].tolist() == [2.5, 0.0]
    assert bets["profit"].tolist() == [1.5, -1.0]


def test_build_bet_simulation_report_summarizes_basic_counts_and_profit():
    bets = simulate_flat_bets(_sample_opportunities(), stake=1.0)
    report = build_bet_simulation_report(bets)

    assert report.loc[0, "n_bets"] == 2
    assert report.loc[0, "n_wins"] == 1
    assert report.loc[0, "n_losses"] == 1
    assert report.loc[0, "total_staked"] == 2.0
    assert report.loc[0, "total_return"] == 2.5
    assert report.loc[0, "total_profit"] == 0.5


def test_simulate_flat_bets_requires_expected_columns():
    opportunities = _sample_opportunities().drop(columns=["selected"])

    with pytest.raises(ValueError, match="selected"):
        simulate_flat_bets(opportunities)
