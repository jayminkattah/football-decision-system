from __future__ import annotations

from fbsystem.backtest.summary import save_backtest_outputs


def main() -> None:
    paths = save_backtest_outputs()

    print("Walk-forward backtest summary build complete.")
    print(f"Season summary: {paths['by_season']}")
    print(f"Cumulative backtest: {paths['cumulative']}")
    print(f"Backtest report: {paths['report']}")


if __name__ == "__main__":
    main()
