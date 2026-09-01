# dashboard/components/metrics.py

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


def _pretty_name(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").title()


def _format_metric_value(name: str, value: Any) -> str:
    if pd.isna(value):
        return "N/A"

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)

    lower_name = name.lower()

    if any(token in lower_name for token in ["roi", "rate", "accuracy", "yield"]):
        return f"{numeric_value:.2%}"

    if any(token in lower_name for token in ["profit", "pnl", "stake", "return"]):
        return f"{numeric_value:,.2f}"

    if numeric_value.is_integer():
        return f"{numeric_value:,.0f}"

    return f"{numeric_value:,.4f}"


def render_metric_cards(
    metrics: dict[str, Any],
    title: str,
    preferred_order: list[str] | None = None,
) -> None:
    st.subheader(title)

    if not metrics:
        st.info("No metrics available yet.")
        return

    if preferred_order:
        ordered_keys = [key for key in preferred_order if key in metrics]
        remaining_keys = [key for key in metrics if key not in ordered_keys]
        keys = ordered_keys + remaining_keys
    else:
        keys = list(metrics.keys())

    cols = st.columns(min(4, len(keys)))

    for idx, key in enumerate(keys):
        with cols[idx % len(cols)]:
            st.metric(
                label=_pretty_name(str(key)),
                value=_format_metric_value(str(key), metrics[key]),
            )


def render_dataframe_summary(df: pd.DataFrame, title: str) -> None:
    st.subheader(title)

    if df.empty:
        st.info("No data available.")
        return

    cols = st.columns(3)

    with cols[0]:
        st.metric("Rows", f"{len(df):,}")

    with cols[1]:
        st.metric("Columns", f"{df.shape[1]:,}")

    with cols[2]:
        numeric_cols = df.select_dtypes(include="number").columns
        st.metric("Numeric Columns", f"{len(numeric_cols):,}")