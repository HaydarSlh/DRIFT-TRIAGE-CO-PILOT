"""Queue & DLQ — depth, recent jobs, dead-letter management."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

import api_client
from auth import require_role


def render() -> None:
    st.markdown("## ⚙️ Queue & DLQ")
    st.caption(
        "Jobs are dispatched onto the Redis-backed task queue with deterministic "
        "idempotency keys. Failures retry with exponential backoff, then land in "
        "the DLQ for triage."
    )

    # ------------------------------------------------------------------
    # Top metrics
    # ------------------------------------------------------------------
    try:
        metrics = api_client.agent_metrics()
        dlq_size = metrics.get("dlq_size", 0)
        worker_pool = metrics.get("worker_pool", {})
    except api_client.ApiError as e:
        st.error(f"Could not load metrics: {e}")
        metrics = {}
        dlq_size = 0
        worker_pool = {}

    col1, col2, col3 = st.columns(3)
    col1.metric("DLQ depth", dlq_size)
    col2.metric("Worker concurrency", worker_pool.get("configured_concurrency", "—"))
    col3.metric("Now (UTC)", datetime.now(timezone.utc).strftime("%H:%M:%S"))

    st.divider()

    # ------------------------------------------------------------------
    # DLQ list
    # ------------------------------------------------------------------
    st.markdown("### Dead-letter queue")
    try:
        dlq_data = api_client.get_dlq(limit=100)
        entries = dlq_data.get("entries", [])
    except api_client.ApiError as e:
        st.error(f"Failed to load DLQ: {e}")
        entries = []

    if not entries:
        st.success("DLQ is empty.")
    else:
        df = pd.DataFrame(entries)
        if "enqueued_at" in df.columns:
            df["enqueued_at"] = pd.to_datetime(df["enqueued_at"]).dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        st.dataframe(
            df[[c for c in ["job_id", "task_name", "attempts", "error", "enqueued_at"] if c in df.columns]],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Triage")
        chosen = st.selectbox(
            "Select a job for action",
            [e["job_id"] for e in entries],
            format_func=lambda j: f"{j[:12]} — {next((e['task_name'] for e in entries if e['job_id'] == j), '?')}",
        )

        col_a, col_b, col_c = st.columns([1, 1, 2])
        if col_a.button("🔁 Requeue"):
            if not require_role("admin"):
                return
            try:
                res = api_client.requeue_dlq(chosen)
                st.success(f"Requeued: {res}")
                st.cache_data.clear()
                st.rerun()
            except api_client.ApiError as e:
                st.error(f"Requeue failed: {e}")
        if col_b.button("🗑️ Delete"):
            if not require_role("admin"):
                return
            try:
                api_client.delete_dlq(chosen)
                st.success("Entry removed.")
                st.cache_data.clear()
                st.rerun()
            except api_client.ApiError as e:
                st.error(f"Delete failed: {e}")

        with st.expander("Bulk purge"):
            older_than = st.number_input(
                "Older than (seconds)", min_value=0, value=86400, step=60,
                help="Use 0 to purge everything."
            )
            if st.button("⚠️ Purge"):
                if not require_role("admin"):
                    return
                try:
                    res = api_client.purge_dlq(older_than_seconds=older_than or None)
                    st.success(f"Purged: {res}")
                    st.cache_data.clear()
                    st.rerun()
                except api_client.ApiError as e:
                    st.error(f"Purge failed: {e}")

    st.divider()

    # ------------------------------------------------------------------
    # Job lookup
    # ------------------------------------------------------------------
    st.markdown("### Inspect a job by ID")
    job_id = st.text_input("Job ID", placeholder="e.g. abc123def456")
    if job_id:
        try:
            job = api_client.get_job(job_id)
            st.json(job)
        except api_client.ApiError as e:
            st.error(f"Lookup failed: {e}")
