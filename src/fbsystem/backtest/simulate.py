"""Simple Day 8 bet simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fbsystem.staking.flat import DEFAULT_STAKE, apply_flat_stakes


REQUIRED_COLUMNS = [
    "match_id",
    "match_date",
    "season",
    "season_start_year",
    "source_country",
    "source_league_code",
    "league",
    "home_team",
    "away_team",
    "full_time_result",
    "outcome",
    "outcome_label",
    "odds",
    "market_prob",
    "model_prob",
    "edge",
    "selected",
]

BET_OUTPUT_COLUMNS = [
    *REQUIRED_COLUMNS,
    "stake",
    "is_win",
    "return",
    "profit",
]


def validate_simulation_input(opportunities: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(opportunities.columns))
    if missing:
        raise ValueError(f"Missing required simulation input columns: {missing}")


def simulate_flat_bets(opportunities: pd.DataFrame, stake: float = DEFAULT_STAKE) -> pd.DataFrame:
    """Simulate one flat-stake bet for each selected opportunity."""
    validate_simulation_input(opportunities)

    bets = opportunities.loc[opportunities["selected"].astype(bool)].copy()
    bets = apply_flat_stakes(bets, stake=stake)

    if bets.empty:
        return pd.DataFrame(columns=BET_OUTPUT_COLUMNS)

    bets["odds"] = pd.to_numeric(bets["odds"], errors="coerce")

    if bets["odds"].isna().any():
        raise ValueError("Selected bets contain missing or non-numeric odds.")

    bets["is_win"] = bets["outcome"].eq(bets["full_time_result"])
    bets["return"] = np.where(bets["is_win"], bets["stake"] * bets["odds"], 0.0)
    bets["profit"] = np.where(
        bets["is_win"],
        bets["stake"] * (bets["odds"] - 1.0),
        -bets["stake"],
    )

    bets["is_win"] = bets["is_win"].astype(bool)

    return bets[BET_OUTPUT_COLUMNS].reset_index(drop=True)


def build_bet_simulation_report(bets: pd.DataFrame) -> pd.DataFrame:
    """Create a compact Day 8 simulation report.

    No ROI or advanced backtest metrics here. Just enough to verify the simulation.
    """
    required = {"stake", "is_win", "return", "profit"}
    missing = sorted(required - set(bets.columns))
    if missing:
        raise ValueError(f"Missing required bet report columns: {missing}")

    n_bets = len(bets)
    n_wins = int(bets["is_win"].sum()) if n_bets else 0

    report = {
        "n_bets": int(n_bets),
        "n_wins": n_wins,
        "n_losses": int(n_bets - n_wins),
        "hit_rate": float(n_wins / n_bets) if n_bets else np.nan,
        "total_staked": float(bets["stake"].sum()) if n_bets else 0.0,
        "total_return": float(bets["return"].sum()) if n_bets else 0.0,
        "total_profit": float(bets["profit"].sum()) if n_bets else 0.0,
    }

    return pd.DataFrame([report])
