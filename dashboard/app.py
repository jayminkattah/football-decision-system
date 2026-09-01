# dashboard/app.py

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st


DASHBOARD_DIR = Path(__file__).resolve().parent
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.append(str(DASHBOARD_DIR))


from components.charts import (  # noqa: E402
    render_backtest_tables,
    render_cumulative_profit_chart,
    render_probability_summary_chart,
    render_roi_by_season_chart,
)
from components.metrics import (  # noqa: E402
    render_dataframe_summary,
    render_metric_cards,
)
from utils.load_data import (  # noqa: E402
    artifact_status,
    existing_figure_paths,
    load_dashboard_data,
    metrics_to_dict,
)


st.set_page_config(
    page_title="Market-Calibrated Probabilistic Decision System",
    page_icon="📊",
    layout="wide",
)


def _select_display_columns(df: pd.DataFrame) -> list[str]:
    preferred_tokens = [
        "date",
        "season",
        "league",
        "home",
        "away",
        "team",
        "selection",
        "market",
        "odds",
        "prob",
        "edge",
        "stake",
        "profit",
        "pnl",
        "result",
    ]

    selected = []

    for column in df.columns:
        lower_col = column.lower()
        if any(token in lower_col for token in preferred_tokens):
            selected.append(column)

    if selected:
        return selected[:16]

    return list(df.columns[:12])


def _render_artifact_health() -> None:
    st.sidebar.header("Artifact Status")

    status_df = artifact_status()
    missing_count = int((~status_df["exists"]).sum())

    if missing_count == 0:
        st.sidebar.success("All expected artifacts found.")
    else:
        st.sidebar.warning(f"{missing_count} artifact(s) missing.")

    with st.sidebar.expander("View artifact paths"):
        st.dataframe(status_df, use_container_width=True, hide_index=True)


def _render_project_overview() -> None:
    st.title("Market-Calibrated Probabilistic Decision System")

    st.markdown(
        """
        This dashboard presents a lean, portfolio-ready view of a football-market
        forecasting and capital allocation system.

        The app reads saved artifacts only. It does not train models, rebuild
        predictions, recalculate strategy signals, or secretly become a second
        pipeline wearing a Streamlit hat.
        """
    )

    st.markdown(
        """
        **Core idea:** compare model-estimated probabilities against market-implied
        probabilities, identify positive-edge opportunities, simulate flat-stake
        decisions, and evaluate walk-forward performance.
        """
    )


def _render_model_section(data: dict[str, pd.DataFrame]) -> None:
    st.header("1. Model Performance")

    model_metrics = metrics_to_dict(data["model_metrics"])

    render_metric_cards(
        model_metrics,
        "Final Model Metrics",
        preferred_order=[
            "log_loss",
            "brier_score",
            "accuracy",
            "n_matches",
            "num_matches",
        ],
    )

    render_probability_summary_chart(data["matches"])

    figure_paths = existing_figure_paths()
    calibration_path = figure_paths.get("model_calibration_figure")

    if calibration_path:
        st.subheader("Basic Calibration Figure")
        st.image(str(calibration_path), use_container_width=True)

    with st.expander("Preview prediction artifact"):
        matches = data["matches"]

        if matches.empty:
            st.info("No prediction data available.")
        else:
            display_cols = _select_display_columns(matches)
            st.dataframe(
                matches[display_cols].head(50),
                use_container_width=True,
            )


def _render_strategy_section(data: dict[str, pd.DataFrame]) -> None:
    st.header("2. Strategy Summary")

    strategy_metrics = metrics_to_dict(data["strategy_metrics"])

    render_metric_cards(
        strategy_metrics,
        "Final Strategy Metrics",
        preferred_order=[
            "n_opportunities",
            "num_opportunities",
            "avg_edge",
            "mean_edge",
            "min_edge",
            "max_edge",
        ],
    )

    opportunities = data["opportunities"]

    render_dataframe_summary(opportunities, "Strategy Opportunities Artifact")

    if opportunities.empty:
        return

    display_cols = _select_display_columns(opportunities)

    edge_cols = [
        column
        for column in opportunities.columns
        if "edge" in column.lower()
        and pd.api.types.is_numeric_dtype(opportunities[column])
    ]

    st.subheader("Sample Strategy Opportunities")

    if edge_cols:
        edge_col = edge_cols[0]
        preview = opportunities.sort_values(edge_col, ascending=False).head(25)
    else:
        preview = opportunities.head(25)

    st.dataframe(
        preview[display_cols],
        use_container_width=True,
    )


def _render_backtest_section(data: dict[str, pd.DataFrame]) -> None:
    st.header("3. Backtest Performance")

    backtest_metrics = metrics_to_dict(data["backtest_metrics"])

    render_metric_cards(
        backtest_metrics,
        "Final Backtest Metrics",
        preferred_order=[
            "total_profit",
            "roi",
            "total_stake",
            "num_bets",
            "hit_rate",
            "max_drawdown",
        ],
    )

    render_cumulative_profit_chart(data["backtest_cumulative"])
    render_roi_by_season_chart(data["backtest_by_season"])

    figure_paths = existing_figure_paths()

    with st.expander("Saved static figures"):
        cumulative_path = figure_paths.get("cumulative_profit_figure")
        roi_path = figure_paths.get("roi_by_season_figure")

        if cumulative_path:
            st.markdown("**Saved cumulative profit figure**")
            st.image(str(cumulative_path), use_container_width=True)

        if roi_path:
            st.markdown("**Saved ROI by season figure**")
            st.image(str(roi_path), use_container_width=True)

        if not cumulative_path and not roi_path:
            st.info("No saved static backtest figures found.")

    render_backtest_tables(
        data["backtest_by_season"],
        data["backtest_cumulative"],
    )


def _render_bets_section(data: dict[str, pd.DataFrame]) -> None:
    st.header("4. Selected / Simulated Bets")

    simulated_bets = data["simulated_bets"]

    if simulated_bets.empty:
        st.info("No simulated bets available.")
        return

    render_dataframe_summary(simulated_bets, "Simulated Bets Artifact")

    display_cols = _select_display_columns(simulated_bets)

    date_cols = [
        column
        for column in simulated_bets.columns
        if "date" in column.lower()
    ]

    if date_cols:
        date_col = date_cols[0]
        preview = simulated_bets.sort_values(date_col, ascending=False).head(50)
    else:
        preview = simulated_bets.tail(50)

    st.subheader("Recent Simulated Bets")
    st.dataframe(
        preview[display_cols],
        use_container_width=True,
    )


def _render_limitations() -> None:
    st.header("5. Limitations")

    st.markdown(
        """
        - This is a research and portfolio project, not a production betting tool.
        - Backtest results depend on historical market prices and available closing/opening odds.
        - Flat staking is intentionally simple; it keeps the evaluation readable.
        - The dashboard does not retrain models or regenerate predictions.
        - Positive historical ROI does not guarantee future profitability. Markets are annoyingly good at becoming less dumb once you notice an edge.
        - Transaction costs, liquidity limits, odds movement, account restrictions, and execution risk are not fully modeled here.
        """
    )


def main() -> None:
    _render_artifact_health()

    data = load_dashboard_data()

    _render_project_overview()

    st.divider()

    _render_model_section(data)

    st.divider()

    _render_strategy_section(data)

    st.divider()

    _render_backtest_section(data)

    st.divider()

    _render_bets_section(data)

    st.divider()

    _render_limitations()


if __name__ == "__main__":
    main()