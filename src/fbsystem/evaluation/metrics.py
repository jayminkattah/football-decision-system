from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from fbsystem.backtest.summary import calculate_max_drawdown


MODEL_PROBABILITY_SETS = [
    ("model_home_prob", "model_draw_prob", "model_away_prob"),
    ("pred_home_prob", "pred_draw_prob", "pred_away_prob"),
    ("pred_home_win_prob", "pred_draw_prob", "pred_away_win_prob"),
    ("pred_home_win", "pred_draw", "pred_away_win"),
    ("home_win_pred", "draw_pred", "away_win_pred"),
    ("home_model_prob", "draw_model_prob", "away_model_prob"),
    ("p_home_model", "p_draw_model", "p_away_model"),
]

MARKET_PROBABILITY_SETS = [
    ("market_home_prob", "market_draw_prob", "market_away_prob"),
    ("market_home_win_prob", "market_draw_prob", "market_away_win_prob"),
    ("home_market_prob", "draw_market_prob", "away_market_prob"),
    ("home_implied_prob", "draw_implied_prob", "away_implied_prob"),
    ("implied_home_prob", "implied_draw_prob", "implied_away_prob"),
    ("p_home_market", "p_draw_market", "p_away_market"),
]

OUTCOME_COLUMNS = [
    "actual_outcome",
    "outcome",
    "result",
    "match_result",
    "full_time_result",
    "target",
]

SCORE_COLUMN_SETS = [
    ("home_goals", "away_goals"),
    ("home_score", "away_score"),
    ("fthg", "ftag"),
]

EDGE_COLUMNS = ["edge", "expected_edge", "value_edge"]
PRED_SELECTED_PROB_COLUMNS = [
    "predicted_probability",
    "model_probability",
    "selected_model_probability",
    "selection_model_probability",
]
MARKET_SELECTED_PROB_COLUMNS = [
    "market_probability",
    "implied_probability",
    "selected_market_probability",
    "selection_market_probability",
]
STAKE_COLUMNS = ["stake", "stake_size", "amount_staked"]
RETURN_COLUMNS = ["total_return", "return", "returns", "payout", "settled_return"]
PROFIT_COLUMNS = ["profit", "pnl", "net_profit"]
ODDS_COLUMNS = ["odds", "decimal_odds", "selected_odds", "bet_odds"]


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


def _find_probability_set(
    df: pd.DataFrame,
    candidates: list[tuple[str, str, str]],
    label: str,
    required: bool = True,
) -> tuple[str, str, str] | None:
    for cols in candidates:
        if all(col in df.columns for col in cols):
            return cols

    if required:
        raise ValueError(
            f"Could not find {label} probability columns. Tried: {candidates}. "
            f"Available columns: {list(df.columns)}"
        )

    return None


def _outcome_labels(df: pd.DataFrame) -> pd.Series:
    outcome_col = _find_column(df, OUTCOME_COLUMNS, required=False)

    if outcome_col is not None:
        raw = df[outcome_col]

        if pd.api.types.is_numeric_dtype(raw):
            labels = pd.to_numeric(raw, errors="coerce")
            return labels.where(labels.isin([0, 1, 2]))

        normalized = raw.astype(str).str.strip().str.lower()

        mapping = {
            "h": 0,
            "home": 0,
            "home_win": 0,
            "home win": 0,
            "1": 0,
            "d": 1,
            "draw": 1,
            "x": 1,
            "a": 2,
            "away": 2,
            "away_win": 2,
            "away win": 2,
            "2": 2,
        }

        return normalized.map(mapping)

    for home_col, away_col in SCORE_COLUMN_SETS:
        if home_col in df.columns and away_col in df.columns:
            home_goals = pd.to_numeric(df[home_col], errors="coerce")
            away_goals = pd.to_numeric(df[away_col], errors="coerce")

            return pd.Series(
                np.select(
                    [
                        home_goals > away_goals,
                        home_goals == away_goals,
                        home_goals < away_goals,
                    ],
                    [0, 1, 2],
                    default=np.nan,
                ),
                index=df.index,
            )

    raise ValueError(
        "Could not infer match outcome. Need one outcome column or home/away score columns."
    )


def _safe_probability_matrix(values: pd.DataFrame) -> np.ndarray:
    probs = values.astype(float).to_numpy()
    probs = np.clip(probs, 1e-15, 1.0)

    row_sums = probs.sum(axis=1)
    valid = row_sums > 0

    probs[valid] = probs[valid] / row_sums[valid, None]

    return probs


def _multiclass_brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    one_hot = np.eye(3)[y_true]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def _accuracy_from_probabilities(y_true: np.ndarray, probs: np.ndarray) -> float:
    predictions = probs.argmax(axis=1)
    return float((predictions == y_true).mean())


def build_model_metrics(matches_with_predictions: pd.DataFrame) -> pd.DataFrame:
    model_cols = _find_probability_set(
        matches_with_predictions,
        MODEL_PROBABILITY_SETS,
        label="model",
    )
    market_cols = _find_probability_set(
        matches_with_predictions,
        MARKET_PROBABILITY_SETS,
        label="market",
        required=False,
    )

    y = _outcome_labels(matches_with_predictions)

    needed_cols = list(model_cols)
    if market_cols is not None:
        needed_cols += list(market_cols)

    valid_mask = y.notna()
    for col in needed_cols:
        valid_mask &= matches_with_predictions[col].notna()

    df = matches_with_predictions.loc[valid_mask].copy()
    y_valid = y.loc[valid_mask].astype(int).to_numpy()

    model_raw = df[list(model_cols)].astype(float)
    model_probs = _safe_probability_matrix(model_raw)

    model_metrics = {
        "n_matches": int(len(df)),
        "model_log_loss": float(log_loss(y_valid, model_probs, labels=[0, 1, 2])),
        "model_brier_score": _multiclass_brier_score(y_valid, model_probs),
        "model_accuracy": _accuracy_from_probabilities(y_valid, model_probs),
        "model_probability_sum_min": float(model_raw.sum(axis=1).min()),
        "model_probability_sum_max": float(model_raw.sum(axis=1).max()),
        "mean_model_confidence": float(model_probs.max(axis=1).mean()),
    }

    if market_cols is not None:
        market_raw = df[list(market_cols)].astype(float)
        market_probs = _safe_probability_matrix(market_raw)

        market_log_loss = float(log_loss(y_valid, market_probs, labels=[0, 1, 2]))
        market_brier = _multiclass_brier_score(y_valid, market_probs)

        model_metrics.update(
            {
                "market_log_loss": market_log_loss,
                "market_brier_score": market_brier,
                "market_accuracy": _accuracy_from_probabilities(y_valid, market_probs),
                "mean_market_confidence": float(market_probs.max(axis=1).mean()),
                "mean_abs_model_market_probability_difference": float(
                    np.abs(model_probs - market_probs).mean()
                ),
                "model_log_loss_minus_market_log_loss": float(
                    model_metrics["model_log_loss"] - market_log_loss
                ),
                "model_brier_minus_market_brier": float(
                    model_metrics["model_brier_score"] - market_brier
                ),
            }
        )
    else:
        model_metrics.update(
            {
                "market_log_loss": np.nan,
                "market_brier_score": np.nan,
                "market_accuracy": np.nan,
                "mean_market_confidence": np.nan,
                "mean_abs_model_market_probability_difference": np.nan,
                "model_log_loss_minus_market_log_loss": np.nan,
                "model_brier_minus_market_brier": np.nan,
            }
        )

    return pd.DataFrame([model_metrics])


def build_calibration_table(
    matches_with_predictions: pd.DataFrame,
    n_bins: int = 10,
) -> pd.DataFrame:
    model_cols = _find_probability_set(
        matches_with_predictions,
        MODEL_PROBABILITY_SETS,
        label="model",
    )

    y = _outcome_labels(matches_with_predictions)

    valid_mask = y.notna()
    for col in model_cols:
        valid_mask &= matches_with_predictions[col].notna()

    df = matches_with_predictions.loc[valid_mask].copy()
    y_valid = y.loc[valid_mask].astype(int).to_numpy()

    model_probs = _safe_probability_matrix(df[list(model_cols)])

    rows = []
    class_names = ["home", "draw", "away"]

    for class_id, class_name in enumerate(class_names):
        class_df = pd.DataFrame(
            {
                "class_name": class_name,
                "predicted_probability": model_probs[:, class_id],
                "observed": (y_valid == class_id).astype(int),
            }
        )
        rows.append(class_df)

    calibration = pd.concat(rows, ignore_index=True)
    calibration["bin"] = pd.cut(
        calibration["predicted_probability"],
        bins=np.linspace(0.0, 1.0, n_bins + 1),
        include_lowest=True,
    )

    grouped = (
        calibration.groupby("bin", observed=False)
        .agg(
            n_predictions=("observed", "size"),
            mean_predicted_probability=("predicted_probability", "mean"),
            observed_rate=("observed", "mean"),
        )
        .reset_index()
    )

    grouped["bin_left"] = grouped["bin"].apply(lambda interval: interval.left)
    grouped["bin_right"] = grouped["bin"].apply(lambda interval: interval.right)
    grouped["absolute_calibration_error"] = (
        grouped["observed_rate"] - grouped["mean_predicted_probability"]
    ).abs()

    return grouped[
        [
            "bin_left",
            "bin_right",
            "n_predictions",
            "mean_predicted_probability",
            "observed_rate",
            "absolute_calibration_error",
        ]
    ]


def build_strategy_metrics(
    strategy_opportunities: pd.DataFrame,
    simulated_bets: pd.DataFrame,
) -> pd.DataFrame:
    metrics: dict[str, float | int] = {
        "n_strategy_opportunities": int(len(strategy_opportunities)),
        "n_bets": int(len(simulated_bets)),
    }

    edge_col = _find_column(strategy_opportunities, EDGE_COLUMNS, required=False)
    pred_prob_col = _find_column(strategy_opportunities, PRED_SELECTED_PROB_COLUMNS, required=False)
    market_prob_col = _find_column(strategy_opportunities, MARKET_SELECTED_PROB_COLUMNS, required=False)

    if edge_col is not None and len(strategy_opportunities) > 0:
        edge = pd.to_numeric(strategy_opportunities[edge_col], errors="coerce")
        metrics.update(
            {
                "mean_edge": float(edge.mean()),
                "median_edge": float(edge.median()),
                "max_edge": float(edge.max()),
            }
        )
    else:
        metrics.update({"mean_edge": np.nan, "median_edge": np.nan, "max_edge": np.nan})

    if pred_prob_col is not None and len(strategy_opportunities) > 0:
        metrics["mean_selected_model_probability"] = float(
            pd.to_numeric(strategy_opportunities[pred_prob_col], errors="coerce").mean()
        )
    else:
        metrics["mean_selected_model_probability"] = np.nan

    if market_prob_col is not None and len(strategy_opportunities) > 0:
        metrics["mean_selected_market_probability"] = float(
            pd.to_numeric(strategy_opportunities[market_prob_col], errors="coerce").mean()
        )
    else:
        metrics["mean_selected_market_probability"] = np.nan

    if simulated_bets.empty:
        metrics.update(
            {
                "total_staked": 0.0,
                "total_return": 0.0,
                "profit": 0.0,
                "roi": 0.0,
                "win_rate": 0.0,
                "average_odds": np.nan,
                "max_drawdown": 0.0,
            }
        )
        return pd.DataFrame([metrics])

    stake_col = _find_column(simulated_bets, STAKE_COLUMNS)
    return_col = _find_column(simulated_bets, RETURN_COLUMNS, required=False)
    profit_col = _find_column(simulated_bets, PROFIT_COLUMNS, required=False)
    odds_col = _find_column(simulated_bets, ODDS_COLUMNS, required=False)

    stake = pd.to_numeric(simulated_bets[stake_col], errors="coerce").fillna(0.0)

    if return_col is not None:
        total_return_series = pd.to_numeric(simulated_bets[return_col], errors="coerce").fillna(0.0)
    elif profit_col is not None:
        total_return_series = stake + pd.to_numeric(simulated_bets[profit_col], errors="coerce").fillna(0.0)
    else:
        raise ValueError("Could not compute strategy returns. Need return or profit column.")

    if profit_col is not None:
        profit_series = pd.to_numeric(simulated_bets[profit_col], errors="coerce").fillna(0.0)
    else:
        profit_series = total_return_series - stake

    total_staked = float(stake.sum())
    total_return = float(total_return_series.sum())
    profit = float(profit_series.sum())

    metrics.update(
        {
            "total_staked": total_staked,
            "total_return": total_return,
            "profit": profit,
            "roi": profit / total_staked if total_staked > 0 else 0.0,
            "win_rate": float((profit_series > 0).mean()),
            "average_odds": float(pd.to_numeric(simulated_bets[odds_col], errors="coerce").mean())
            if odds_col is not None
            else np.nan,
            "max_drawdown": calculate_max_drawdown(profit_series),
        }
    )

    return pd.DataFrame([metrics])


def build_final_backtest_metrics(
    backtest_by_season: pd.DataFrame,
    backtest_cumulative: pd.DataFrame,
) -> pd.DataFrame:
    if backtest_by_season.empty or backtest_cumulative.empty:
        return pd.DataFrame(
            [
                {
                    "n_seasons": 0,
                    "profitable_seasons": 0,
                    "profitable_season_rate": 0.0,
                    "best_season": None,
                    "worst_season": None,
                    "best_season_roi": np.nan,
                    "worst_season_roi": np.nan,
                    "mean_season_roi": np.nan,
                    "season_roi_std": np.nan,
                    "final_cumulative_profit": 0.0,
                    "max_drawdown": 0.0,
                }
            ]
        )

    by_season = backtest_by_season.copy()
    cumulative = backtest_cumulative.copy()

    best_idx = by_season["roi"].idxmax()
    worst_idx = by_season["roi"].idxmin()

    metrics = {
        "n_seasons": int(len(by_season)),
        "profitable_seasons": int((by_season["profit"] > 0).sum()),
        "profitable_season_rate": float((by_season["profit"] > 0).mean()),
        "best_season": by_season.loc[best_idx, "season"],
        "worst_season": by_season.loc[worst_idx, "season"],
        "best_season_roi": float(by_season.loc[best_idx, "roi"]),
        "worst_season_roi": float(by_season.loc[worst_idx, "roi"]),
        "mean_season_roi": float(by_season["roi"].mean()),
        "season_roi_std": float(by_season["roi"].std(ddof=0)),
        "final_cumulative_profit": float(cumulative["cumulative_profit"].iloc[-1]),
        "max_drawdown": float(cumulative["drawdown"].max()),
    }

    return pd.DataFrame([metrics])


def save_cumulative_profit_figure(backtest_cumulative: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cumulative = backtest_cumulative.copy()
    cumulative["date"] = pd.to_datetime(cumulative["date"])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cumulative["date"], cumulative["cumulative_profit"])
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title("Cumulative Profit")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Profit")
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_roi_by_season_figure(backtest_by_season: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    by_season = backtest_by_season.copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(by_season["season"], by_season["roi"])
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title("ROI by Season")
    ax.set_xlabel("Season")
    ax.set_ylabel("ROI")
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_model_calibration_figure(
    calibration_table: pd.DataFrame,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    table = calibration_table.dropna(
        subset=["mean_predicted_probability", "observed_rate"]
    ).copy()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.scatter(table["mean_predicted_probability"], table["observed_rate"])
    ax.set_title("Basic Model Calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
