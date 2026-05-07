"""Investigations — list view + drill-down with audit trail and final state."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import api_client


SEVERITY_ICON = {"green": "🟢", "yellow": "🟡", "amber": "🟠", "red": "🔴"}
STATUS_ICON = {
    "pending": "⏳",
    "running": "⚙️",
    "awaiting_approval": "✋",
    "completed": "✅",
    "failed": "❌",
}


def render() -> None:
    st.markdown("## 🔍 Investigations")
    st.caption("Every drift webhook becomes one investigation. Click into a row for details.")

    try:
        invs = api_client.list_investigations()
    except api_client.ApiError as e:
        st.error(f"Failed to fetch investigations: {e}")
        return

    if not invs:
        st.info("No investigations yet.")
        return

    # Filters
    col1, col2, col3 = st.columns([2, 2, 1])
    statuses = sorted({i.get("status", "") for i in invs})
    severities = sorted({i.get("severity", "") for i in invs})
    f_status = col1.multiselect("Status", statuses, default=statuses)
    f_sev = col2.multiselect("Severity", severities, default=severities)
    search = col3.text_input("Search model_id", "")

    filtered = [
        i
        for i in invs
        if i.get("status") in f_status
        and i.get("severity") in f_sev
        and (search.lower() in (i.get("model_id", "")).lower() if search else True)
    ]

    df = pd.DataFrame(filtered)
    if df.empty:
        st.info("No investigations match the current filters.")
        return

    # Pretty render: glyph icons + truncated IDs
    df_display = df.copy()
    if "severity" in df_display.columns:
        df_display["severity"] = df_display["severity"].map(
            lambda s: f"{SEVERITY_ICON.get(s, '•')} {s}"
        )
    if "status" in df_display.columns:
        df_display["status"] = df_display["status"].map(
            lambda s: f"{STATUS_ICON.get(s, '•')} {s}"
        )
    if "id" in df_display.columns:
        df_display["id_short"] = df_display["id"].str.slice(0, 8)
    if "created_at" in df_display.columns:
        df_display["created_at"] = pd.to_datetime(df_display["created_at"]).dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    cols_to_show = [
        c
        for c in ["id_short", "created_at", "model_id", "severity", "status", "drift_event_id"]
        if c in df_display.columns
    ]
    st.dataframe(df_display[cols_to_show], use_container_width=True, hide_index=True)

    st.divider()

    # Drill-down selector
    options = {f"{i['id'][:8]} — {i.get('model_id', '?')} — {i.get('status', '?')}": i["id"] for i in filtered}
    chosen_label = st.selectbox("Inspect investigation", list(options.keys()))
    if chosen_label:
        _render_detail(options[chosen_label])


def _render_detail(inv_id: str) -> None:
    try:
        detail = api_client.get_investigation(inv_id)
    except api_client.ApiError as e:
        st.error(f"Failed to load investigation: {e}")
        return

    st.markdown(f"### Investigation `{inv_id[:8]}`")

    # Header metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", detail.get("status", "—"))
    c2.metric("Severity", detail.get("severity", "—"))
    c3.metric("Model", detail.get("model_id", "—"))
    c4.metric("Steps", len(detail.get("steps", [])))

    tab_state, tab_trail, tab_action, tab_raw = st.tabs(
        ["Final state", "Audit trail", "Recommended action", "Raw JSON"]
    )

    with tab_state:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Severity classified**")
            st.write(detail.get("severity_classified") or "_pending_")
            st.markdown("**Comms draft**")
            st.text(detail.get("comms_draft") or "_(none)_")
        with col2:
            st.markdown("**Retrieved context**")
            ctx = detail.get("context")
            if ctx:
                st.json(ctx, expanded=False)
            else:
                st.write("_(none captured)_")

    with tab_trail:
        steps = detail.get("steps", [])
        if not steps:
            st.info("No steps recorded yet.")
        else:
            df = pd.DataFrame(steps)
            if "created_at" in df.columns:
                df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            cols = [c for c in ["step_number", "agent_name", "duration_ms", "created_at"] if c in df.columns]
            st.dataframe(df[cols].sort_values("step_number"), use_container_width=True, hide_index=True)

    with tab_action:
        action = detail.get("recommended_action")
        if not action:
            st.info("No action recommended yet.")
        else:
            st.write(f"**Type:** `{action.get('action_type', '—')}`")
            st.write(f"**Target version:** `{action.get('target_version', '—')}`")
            st.write("**Justification:**")
            st.write(action.get("justification", "_(none)_"))

    with tab_raw:
        st.json(detail, expanded=False)
