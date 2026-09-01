from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATE_COLUMNS = ["date", "match_date"]
SEASON_COLUMNS = ["season"]
STAKE_COLUMNS = ["stake", "stake_size", "amount_staked"]
RETURN_COLUMNS = ["total_return", "return", "returns", "payout", "settled_return"]
PROFIT_COLUMNS = ["profit", "pnl", "net_profit"]
ODDS_COLUMNS = ["odds", "decimal_odds", "selected_odds", "bet_odds"]
WIN_COLUMNS = ["won", "bet_won", "is_win"]


def _find_column(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col

    if required:
        raise ValueError(
            f"Could not find any of these required columns: {candidates}. "
            f"Available columns: {list(df.columns)}"
        )

    return None


def _derive_season(dates: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(dates)
    start_year = np.where(parsed.dt.month >= 7, parsed.dt.year, parsed.dt.year - 1)
    end_year = start_year + 1
    return pd.Series(start_year.astype(str) + "-" + end_year.astype(str), index=dates.index)


def _normalise_simulated_bets(bets: pd.DataFrame) -> pd.DataFrame:
    if bets.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "season",
                "stake",
                "total_return",
                "profit",
                "odds",
                "won",
            ]
        )

    df = bets.copy()

    date_col = _find_column(df, DATE_COLUMNS)
    season_col = _find_column(df, SEASON_COLUMNS, required=False)
    stake_col = _find_column(df, STAKE_COLUMNS)
    return_col = _find_column(df, RETURN_COLUMNS, required=False)
    profit_col = _find_column(df, PROFIT_COLUMNS, required=False)
    odds_col = _find_column(df, ODDS_COLUMNS, required=False)
    win_col = _find_column(df, WIN_COLUMNS, required=False)

    out = pd.DataFrame(index=df.index)
    out["date"] = pd.to_datetime(df[date_col])
    out["season"] = df[season_col].astype(str) if season_col else _derive_season(out["date"])
    out["stake"] = pd.to_numeric(df[stake_col], errors="coerce").fillna(0.0)

    if return_col is not None:
        out["total_return"] = pd.to_numeric(df[return_col], errors="coerce").fillna(0.0)
    else:
        out["total_return"] = np.nan

    if profit_col is not None:
        out["profit"] = pd.to_numeric(df[profit_col], errors="coerce").fillna(0.0)
    else:
        out["profit"] = np.nan

    if return_col is None and profit_col is not None:
        out["total_return"] = out["stake"] + out["profit"]

    if profit_col is None and return_col is not None:
        out["profit"] = out["total_return"] - out["stake"]

    if return_col is None and profit_col is None:
        if odds_col is None or win_col is None:
            raise ValueError(
                "Could not compute profit. Need either a return/profit column, "
                "or both odds and win indicator columns."
            )

        odds = pd.to_numeric(df[odds_col], errors="coerce")
        won = df[win_col].astype(bool)
        out["total_return"] = np.where(won, out["stake"] * odds, 0.0)
        out["profit"] = out["total_return"] - out["stake"]

    if odds_col is not None:
        out["odds"] = pd.to_numeric(df[odds_col], errors="coerce")
    else:
        out["odds"] = np.nan

    if win_col is not None:
        out["won"] = df[win_col].astype(bool)
    else:
        out["won"] = out["profit"] > 0

    return out.sort_values(["date", "season"]).reset_index(drop=True)


def calculate_max_drawdown(profit: pd.Series) -> float:
    if profit.empty:
        return 0.0

    cumulative_profit = profit.fillna(0.0).cumsum()
    running_peak = pd.Series(
        np.maximum.accumulate(np.r_[0.0, cumulative_profit.to_numpy()])[:-1],
        index=cumulative_profit.index,
    )
    drawdown = running_peak - cumulative_profit

    return float(drawdown.max())


def summarize_backtest_by_season(bets: pd.DataFrame) -> pd.DataFrame:
    normalised = _normalise_simulated_bets(bets)

    rows: list[dict[str, object]] = []

    for season, group in normalised.groupby("season", sort=True):
        group = group.sort_values("date")

        n_bets = int(len(group))
        total_staked = float(group["stake"].sum())
        total_return = float(group["total_return"].sum())
        profit = float(group["profit"].sum())

        rows.append(
            {
                "season": season,
                "n_bets": n_bets,
                "total_staked": total_staked,
                "total_return": total_return,
                "profit": profit,
                "roi": profit / total_staked if total_staked > 0 else 0.0,
                "win_rate": float(group["won"].mean()) if n_bets > 0 else 0.0,
                "average_odds": float(group["odds"].mean()) if group["odds"].notna().any() else np.nan,
                "max_drawdown": calculate_max_drawdown(group["profit"]),
            }
        )

    return pd.DataFrame(rows)


def build_cumulative_backtest(bets: pd.DataFrame) -> pd.DataFrame:
    normalised = _normalise_simulated_bets(bets)

    if normalised.empty:
        return pd.DataFrame(
            columns=[
                "bet_number",
                "date",
                "season",
                "stake",
                "total_return",
                "profit",
                "cumulative_staked",
                "cumulative_return",
                "cumulative_profit",
                "running_roi",
                "running_peak_profit",
                "drawdown",
            ]
        )

    out = normalised.sort_values("date").reset_index(drop=True)
    out.insert(0, "bet_number", np.arange(1, len(out) + 1))

    out["cumulative_staked"] = out["stake"].cumsum()
    out["cumulative_return"] = out["total_return"].cumsum()
    out["cumulative_profit"] = out["profit"].cumsum()
    out["running_roi"] = np.where(
        out["cumulative_staked"] > 0,
        out["cumulative_profit"] / out["cumulative_staked"],
        0.0,
    )

    out["running_peak_profit"] = np.maximum.accumulate(
        np.r_[0.0, out["cumulative_profit"].to_numpy()]
    )[1:]
    out["drawdown"] = out["running_peak_profit"] - out["cumulative_profit"]

    return out[
        [
            "bet_number",
            "date",
            "season",
            "stake",
            "total_return",
            "profit",
            "cumulative_staked",
            "cumulative_return",
            "cumulative_profit",
            "running_roi",
            "running_peak_profit",
            "drawdown",
        ]
    ]


def build_backtest_report(bets: pd.DataFrame) -> pd.DataFrame:
    by_season = summarize_backtest_by_season(bets)
    cumulative = build_cumulative_backtest(bets)

    if cumulative.empty:
        return pd.DataFrame(
            [
                {
                    "n_bets": 0,
                    "total_staked": 0.0,
                    "total_return": 0.0,
                    "profit": 0.0,
                    "roi": 0.0,
                    "win_rate": 0.0,
                    "average_odds": np.nan,
                    "max_drawdown": 0.0,
                    "n_seasons": 0,
                    "profitable_seasons": 0,
                }
            ]
        )

    total_staked = float(cumulative["stake"].sum())
    total_return = float(cumulative["total_return"].sum())
    profit = float(cumulative["profit"].sum())

    report = {
        "n_bets": int(len(cumulative)),
        "total_staked": total_staked,
        "total_return": total_return,
        "profit": profit,
        "roi": profit / total_staked if total_staked > 0 else 0.0,
        "win_rate": float((cumulative["profit"] > 0).mean()),
        "average_odds": np.nan,
        "max_drawdown": float(cumulative["drawdown"].max()),
        "n_seasons": int(len(by_season)),
        "profitable_seasons": int((by_season["profit"] > 0).sum()) if not by_season.empty else 0,
    }

    return pd.DataFrame([report])


def save_backtest_outputs(
    simulated_bets_path: str | Path = "data/processed/simulated_bets.parquet",
    output_dir: str | Path = "outputs/backtests",
    evaluation_dir: str | Path = "outputs/evaluation",
) -> dict[str, Path]:
    simulated_bets_path = Path(simulated_bets_path)
    output_dir = Path(output_dir)
    evaluation_dir = Path(evaluation_dir)

    if not simulated_bets_path.exists():
        raise FileNotFoundError(f"Missing input file: {simulated_bets_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    bets = pd.read_parquet(simulated_bets_path)

    by_season = summarize_backtest_by_season(bets)
    cumulative = build_cumulative_backtest(bets)
    report = build_backtest_report(bets)

    by_season_path = output_dir / "backtest_by_season.csv"
    cumulative_path = output_dir / "backtest_cumulative.csv"
    report_path = evaluation_dir / "backtest_report.csv"

    by_season.to_csv(by_season_path, index=False)
    cumulative.to_csv(cumulative_path, index=False)
    report.to_csv(report_path, index=False)

    return {
        "by_season": by_season_path,
        "cumulative": cumulative_path,
        "report": report_path,
    }
