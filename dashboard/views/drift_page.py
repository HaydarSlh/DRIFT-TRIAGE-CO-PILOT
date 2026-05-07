"""Drift Monitor — visualize PSI / chi² / output drift, plus a manual predict tester.

The platform doesn't yet expose a `GET /drift/` endpoint that returns the rolling
report, so this page derives signal from:
  - the latest investigation's `severity`
  - a manual /predict invocation (which, server-side, re-runs the rolling window)

When the platform exposes a drift-report read endpoint, swap the placeholder
section for live PSI/chi² values.
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

import api_client


SEVERITY_ICON = {"green": "🟢", "yellow": "🟡", "amber": "🟠", "red": "🔴"}


def render() -> None:
    st.markdown("## 📈 Drift Monitor")
    st.caption(
        "PSI on numerics, chi² on categoricals, output-distribution drift — "
        "computed by the platform on a rolling window and emitted via webhook "
        "when the severity changes."
    )

    tab_history, tab_predict = st.tabs(["Severity history", "Predict tester"])

    with tab_history:
        _render_severity_history()

    with tab_predict:
        _render_predict_tester()


def _render_severity_history() -> None:
    """Severity timeline derived from investigations, since each investigation
    is created on a severity transition by the platform's webhook."""
    try:
        invs = api_client.list_investigations()
    except api_client.ApiError as e:
        st.error(f"Failed to load: {e}")
        return

    if not invs:
        st.info("No drift events captured yet.")
        return

    df = pd.DataFrame(invs).sort_values("created_at")
    df["created_at"] = pd.to_datetime(df["created_at"])
    severity_order = {"green": 0, "yellow": 1, "amber": 2, "red": 3}
    df["severity_score"] = df["severity"].map(severity_order).fillna(0)

    st.markdown("### Severity over time")
    chart_df = df.set_index("created_at")[["severity_score"]]
    st.line_chart(chart_df, use_container_width=True)

    counts = df["severity"].value_counts().reset_index()
    counts.columns = ["severity", "count"]
    counts["severity"] = counts["severity"].map(lambda s: f"{SEVERITY_ICON.get(s, '•')} {s}")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Severity distribution**")
        st.dataframe(counts, use_container_width=True, hide_index=True)
    with col2:
        st.markdown("**Most recent events**")
        recent = df.sort_values("created_at", ascending=False).head(5)
        st.dataframe(
            recent[["created_at", "severity", "model_id", "status"]],
            use_container_width=True,
            hide_index=True,
        )


def _render_predict_tester() -> None:
    st.markdown("### Manual prediction")
    st.caption(
        "Send a synthetic feature row to the platform's `/predict` endpoint. "
        "The platform updates its rolling drift window on every call, so this is "
        "the easiest way to drive the agent during a demo."
    )

    sample_payload = _default_payload()
    raw = st.text_area(
        "Request payload (JSON)",
        value=json.dumps(sample_payload, indent=2),
        height=420,
    )

    if st.button("Send /predict", type="primary"):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
            return
        try:
            result = api_client.predict(payload)
            st.success("Response:")
            st.json(result)
        except api_client.ApiError as e:
            st.error(f"Platform rejected the call: {e}")


def _default_payload() -> dict[str, Any]:
    """A minimally-valid Bank Marketing row.

    Keep this in sync with the platform's PredictionRequest schema if columns
    change. The real demo flow is to drift `euribor3m` and `job` to trigger
    the agent.
    """
    return {
        "age": 35,
        "job": "admin.",
        "marital": "married",
        "education": "university.degree",
        "default": "no",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "day_of_week": "thu",
        "campaign": 1,
        "pdays": 999,
        "previous": 0,
        "poutcome": "nonexistent",
        "emp.var.rate": 1.1,
        "cons.price.idx": 93.994,
        "cons.conf.idx": -36.4,
        "euribor3m": 4.857,
        "nr.employed": 5191.0,
    }
