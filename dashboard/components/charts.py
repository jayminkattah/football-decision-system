# dashboard/components/charts.py

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_lookup = {column.lower(): column for column in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower_lookup:
            return lower_lookup[candidate.lower()]

    return None


def _date_or_index_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "date",
        "match_date",
        "event_date",
        "kickoff_date",
        "season",
        "match_id",
    ]

    return _first_existing_column(df, candidates)


def _numeric_column_containing(df: pd.DataFrame, tokens: list[str]) -> str | None:
    numeric_cols = df.select_dtypes(include="number").columns

    for column in numeric_cols:
        lower_col = column.lower()
        if all(token in lower_col for token in tokens):
            return column

    return None


def render_cumulative_profit_chart(backtest_cumulative: pd.DataFrame) -> None:
    st.subheader("Cumulative Profit")

    if backtest_cumulative.empty:
        st.info("Cumulative backtest output is missing.")
        return

    x_col = _date_or_index_column(backtest_cumulative)

    y_col = _first_existing_column(
        backtest_cumulative,
        [
            "cumulative_profit",
            "cum_profit",
            "cumulative_pnl",
            "cum_pnl",
            "running_profit",
            "bankroll",
        ],
    )

    if y_col is None:
        y_col = _numeric_column_containing(backtest_cumulative, ["profit"])

    if y_col is None:
        st.warning("Could not find a cumulative profit column.")
        st.dataframe(backtest_cumulative, use_container_width=True)
        return

    if x_col is None:
        chart_df = backtest_cumulative.reset_index().rename(columns={"index": "row"})
        x_col = "row"
    else:
        chart_df = backtest_cumulative.copy()

    fig = px.line(
        chart_df,
        x=x_col,
        y=y_col,
        markers=False,
        title="Cumulative Profit Over Time",
    )

    fig.update_layout(
        xaxis_title=x_col.replace("_", " ").title(),
        yaxis_title=y_col.replace("_", " ").title(),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_roi_by_season_chart(backtest_by_season: pd.DataFrame) -> None:
    st.subheader("ROI by Season")

    if backtest_by_season.empty:
        st.info("Season-level backtest output is missing.")
        return

    x_col = _first_existing_column(backtest_by_season, ["season", "Season"])

    y_col = _first_existing_column(
        backtest_by_season,
        [
            "roi",
            "ROI",
            "return_on_investment",
            "season_roi",
            "yield",
        ],
    )

    if y_col is None:
        y_col = _numeric_column_containing(backtest_by_season, ["roi"])

    if x_col is None or y_col is None:
        st.warning("Could not find season and ROI columns.")
        st.dataframe(backtest_by_season, use_container_width=True)
        return

    fig = px.bar(
        backtest_by_season,
        x=x_col,
        y=y_col,
        title="Backtest ROI by Season",
    )

    fig.update_layout(
        xaxis_title="Season",
        yaxis_title=y_col.replace("_", " ").title(),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_probability_summary_chart(matches: pd.DataFrame) -> None:
    st.subheader("Model Probability Summary")

    if matches.empty:
        st.info("Prediction artifact is missing.")
        return

    probability_cols = [
        column
        for column in matches.select_dtypes(include="number").columns
        if (
            ("prob" in column.lower() or column.lower().startswith("p_"))
            and "market" not in column.lower()
            and "edge" not in column.lower()
            and "odds" not in column.lower()
        )
    ]

    if not probability_cols:
        st.warning("Could not find model probability columns.")
        st.dataframe(matches.head(25), use_container_width=True)
        return

    chart_cols = probability_cols[:5]

    long_df = matches[chart_cols].melt(
        var_name="probability_column",
        value_name="probability",
    )

    fig = px.histogram(
        long_df,
        x="probability",
        color="probability_column",
        nbins=30,
        title="Distribution of Model Probability Outputs",
    )

    fig.update_layout(
        xaxis_title="Predicted Probability",
        yaxis_title="Count",
        bargap=0.05,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_backtest_tables(
    backtest_by_season: pd.DataFrame,
    backtest_cumulative: pd.DataFrame,
) -> None:
    st.subheader("Backtest Tables")

    if not backtest_by_season.empty:
        st.markdown("**Backtest by season**")
        st.dataframe(backtest_by_season, use_container_width=True)

    if not backtest_cumulative.empty:
        st.markdown("**Latest cumulative backtest rows**")
        st.dataframe(backtest_cumulative.tail(25), use_container_width=True)

    if backtest_by_season.empty and backtest_cumulative.empty:
        st.info("No backtest tables available.")