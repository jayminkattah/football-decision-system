"""Flat staking utilities."""

from __future__ import annotations

import pandas as pd


DEFAULT_STAKE = 1.0


def apply_flat_stakes(bets: pd.DataFrame, stake: float = DEFAULT_STAKE) -> pd.DataFrame:
    """Assign the same stake to every bet row."""
    if stake <= 0:
        raise ValueError("stake must be greater than 0.")

    staked = bets.copy()
    staked["stake"] = float(stake)

    return staked
