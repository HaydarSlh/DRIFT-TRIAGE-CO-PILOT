"""Overview — system health + headline KPIs across both services."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

import api_client


SEVERITY_COLORS = {
    "green": "#16a34a",
    "yellow": "#eab308",
    "amber": "#f59e0b",
    "red": "#dc2626",
}
STATUS_BADGE = {
    "ok": "✅",
    "degraded": "⚠️",
    "failed": "❌",
    "fail": "❌",
    "disabled": "➖",
}


def _badge(value: str) -> str:
    icon = STATUS_BADGE.get(value, "•")
    return f"{icon} {value}"


def render() -> None:
    st.markdown("## 📊 Overview")
    st.caption("Live snapshot of the platform, agent, and queue.")

    # ------------------------------------------------------------------
    # Health row
    # ------------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    try:
        ag = api_client.agent_health()
        agent_status = ag.get("status", "unknown")
        checks = ag.get("checks", {})
    except api_client.ApiError as e:
        agent_status = "failed"
        checks = {"error": str(e)}

    try:
        api_client.platform_health()
        platform_status = "ok"
    except api_client.ApiError:
        platform_status = "failed"

    try:
        metrics = api_client.agent_metrics()
        dlq_size = metrics.get("dlq_size", 0)
    except api_client.ApiError:
        metrics = {}
        dlq_size = 0

    try:
        approvals = api_client.list_pending_approvals()
        pending_approvals = len(approvals)
    except api_client.ApiError:
        pending_approvals = 0

    col1.metric("Agent service", _badge(agent_status))
    col2.metric("ML platform", _badge(platform_status))
    col3.metric("Pending approvals", pending_approvals, help="HIL inbox depth")
    col4.metric("Dead-letter queue", dlq_size, help="Jobs that exhausted retries")

    st.divider()

    # ------------------------------------------------------------------
    # Investigations summary + recent activity
    # ------------------------------------------------------------------
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("### Recent investigations")
        try:
            invs = api_client.list_investigations()
        except api_client.ApiError as e:
            st.error(f"Failed to fetch investigations: {e}")
            invs = []

        if not invs:
            st.info("No investigations yet. Drift events will appear here.")
        else:
            df = pd.DataFrame(invs)
            df = df[
                [
                    c
                    for c in ["created_at", "model_id", "severity", "status", "drift_event_id"]
                    if c in df.columns
                ]
            ]
            if "created_at" in df.columns:
                df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            st.dataframe(df.head(10), use_container_width=True, hide_index=True)

            # Status breakdown
            if "status" in df.columns:
                counts = df["status"].value_counts().rename_axis("status").reset_index(name="count")
                st.bar_chart(counts.set_index("status"))

    with col_right:
        st.markdown("### Service checks")
        if checks:
            for k, v in checks.items():
                st.write(f"**{k}** — `{v}`")
        else:
            st.write("_No check data available_")

        st.markdown("### Worker pool")
        pool = metrics.get("worker_pool", {}) if metrics else {}
        st.write(
            f"Configured concurrency: **{pool.get('configured_concurrency', '—')}**"
        )

        st.markdown("### Now")
        st.caption(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
