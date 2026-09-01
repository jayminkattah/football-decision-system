from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fbsystem.strategies.rules import (  # noqa: E402
    DEFAULT_MIN_EDGE,
    build_strategy_opportunities,
    build_strategy_report,
)


INPUT_PATH = ROOT / "data" / "processed" / "matches_with_predictions.parquet"
OUTPUT_PATH = ROOT / "data" / "processed" / "strategy_opportunities.parquet"
REPORT_PATH = ROOT / "outputs" / "evaluation" / "strategy_report.csv"


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    matches = pd.read_parquet(INPUT_PATH)

    opportunities = build_strategy_opportunities(matches, min_edge=DEFAULT_MIN_EDGE)
    report = build_strategy_report(opportunities, min_edge=DEFAULT_MIN_EDGE)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    opportunities.to_parquet(OUTPUT_PATH, index=False)
    report.to_csv(REPORT_PATH, index=False)

    print("Strategy opportunities build complete.")
    print(f"Input rows: {len(matches):,}")
    print(f"Opportunity rows: {len(opportunities):,}")
    print(f"Selected opportunities: {int(opportunities['selected'].sum()):,}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
