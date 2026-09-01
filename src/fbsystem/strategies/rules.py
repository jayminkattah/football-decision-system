"""Simple strategy rules for converting predictions into opportunities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_MIN_EDGE = 0.02

BASE_COLUMNS = [
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
]


@dataclass(frozen=True)
class OutcomeSpec:
    outcome: str
    outcome_label: str
    odds_col: str
    market_prob_col: str
    model_prob_col: str
    edge_col: str


OUTCOME_SPECS = (
    OutcomeSpec("H", "Home win", "home_odds", "market_home_prob", "model_home_prob", "home_edge"),
    OutcomeSpec("D", "Draw", "draw_odds", "market_draw_prob", "model_draw_prob", "draw_edge"),
    OutcomeSpec("A", "Away win", "away_odds", "market_away_prob", "model_away_prob", "away_edge"),
)

OUTPUT_COLUMNS = [
    *BASE_COLUMNS,
    "outcome",
    "outcome_label",
    "odds",
    "market_prob",
    "model_prob",
    "edge",
    "selected",
]


def _required_columns(outcome_specs: Iterable[OutcomeSpec] = OUTCOME_SPECS) -> list[str]:
    outcome_columns: list[str] = []
    for spec in outcome_specs:
        outcome_columns.extend([
            spec.odds_col,
            spec.market_prob_col,
            spec.model_prob_col,
            spec.edge_col,
        ])
    return [*BASE_COLUMNS, *outcome_columns]


def validate_strategy_input(matches: pd.DataFrame) -> None:
    """Validate that the wide prediction table contains columns needed by Day 7."""
    missing = sorted(set(_required_columns()) - set(matches.columns))
    if missing:
        raise ValueError(f"Missing required strategy input columns: {missing}")


def build_strategy_opportunities(matches: pd.DataFrame, min_edge: float = DEFAULT_MIN_EDGE) -> pd.DataFrame:
    """Convert wide match predictions into one opportunity row per outcome.

    The Day 7 rule is intentionally simple:

        selected = edge >= min_edge
    """
    if min_edge < 0:
        raise ValueError("min_edge must be greater than or equal to 0.")

    validate_strategy_input(matches)

    frames: list[pd.DataFrame] = []

    for spec in OUTCOME_SPECS:
        frame = matches[
            [
                *BASE_COLUMNS,
                spec.odds_col,
                spec.market_prob_col,
                spec.model_prob_col,
                spec.edge_col,
            ]
        ].copy()

        frame["outcome"] = spec.outcome
        frame["outcome_label"] = spec.outcome_label
        frame["odds"] = pd.to_numeric(frame[spec.odds_col], errors="coerce")
        frame["market_prob"] = pd.to_numeric(frame[spec.market_prob_col], errors="coerce")
        frame["model_prob"] = pd.to_numeric(frame[spec.model_prob_col], errors="coerce")
        frame["edge"] = pd.to_numeric(frame[spec.edge_col], errors="coerce")
        frame["selected"] = frame["edge"].ge(min_edge).fillna(False)

        frames.append(frame[OUTPUT_COLUMNS])

    opportunities = pd.concat(frames, ignore_index=True)
    opportunities = opportunities.sort_values(["match_date", "match_id", "outcome"]).reset_index(drop=True)
    opportunities["selected"] = opportunities["selected"].astype(bool)

    return opportunities


def build_strategy_report(opportunities: pd.DataFrame, min_edge: float = DEFAULT_MIN_EDGE) -> pd.DataFrame:
    """Create a compact Day 7 selection report.

    This is not a returns report. It only describes how many opportunities passed
    the edge rule.
    """
    required = {"match_id", "outcome", "edge", "selected"}
    missing = sorted(required - set(opportunities.columns))
    if missing:
        raise ValueError(f"Missing required strategy report columns: {missing}")

    n_opportunities = len(opportunities)
    n_selected = int(opportunities["selected"].sum())
    selected_edges = opportunities.loc[opportunities["selected"], "edge"]

    selected_by_outcome = (
        opportunities.loc[opportunities["selected"]]
        .groupby("outcome")
        .size()
        .reindex(["H", "D", "A"], fill_value=0)
    )

    report = {
        "min_edge": float(min_edge),
        "n_matches": int(opportunities["match_id"].nunique()),
        "n_opportunities": int(n_opportunities),
        "n_selected": n_selected,
        "selection_rate": float(n_selected / n_opportunities) if n_opportunities else np.nan,
        "avg_selected_edge": float(selected_edges.mean()) if n_selected else np.nan,
        "n_selected_home": int(selected_by_outcome.loc["H"]),
        "n_selected_draw": int(selected_by_outcome.loc["D"]),
        "n_selected_away": int(selected_by_outcome.loc["A"]),
    }

    return pd.DataFrame([report])
