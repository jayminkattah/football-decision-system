from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fbsystem.backtest.simulate import (  # noqa: E402
    build_bet_simulation_report,
    simulate_flat_bets,
)
from fbsystem.staking.flat import DEFAULT_STAKE  # noqa: E402


INPUT_PATH = ROOT / "data" / "processed" / "strategy_opportunities.parquet"
OUTPUT_PATH = ROOT / "data" / "processed" / "simulated_bets.parquet"
REPORT_PATH = ROOT / "outputs" / "evaluation" / "bet_simulation_report.csv"


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    opportunities = pd.read_parquet(INPUT_PATH)

    bets = simulate_flat_bets(opportunities, stake=DEFAULT_STAKE)
    report = build_bet_simulation_report(bets)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    bets.to_parquet(OUTPUT_PATH, index=False)
    report.to_csv(REPORT_PATH, index=False)

    print("Bet simulation build complete.")
    print(f"Opportunity rows: {len(opportunities):,}")
    print(f"Simulated bets: {len(bets):,}")
    print(f"Total staked: {float(bets['stake'].sum()) if len(bets) else 0.0:,.2f}")
    print(f"Total profit: {float(bets['profit'].sum()) if len(bets) else 0.0:,.2f}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
