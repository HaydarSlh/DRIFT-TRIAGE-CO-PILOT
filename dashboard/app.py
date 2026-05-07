"""Drift Triage Co-Pilot — Streamlit dashboard.

Single-process control plane for the operator: registry state, drift health,
agent investigations, queue + DLQ depth, and the HIL approval inbox. Auth
gates everything; the first account that signs up becomes admin.
"""
from __future__ import annotations

import streamlit as st

from auth import current_user, is_authed, logout, render_login_signup
from views import (
    approvals_page,
    drift_page,
    investigations_page,
    overview_page,
    queue_page,
    registry_page,
)

st.set_page_config(
    page_title="Drift Triage Co-Pilot",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Auth gate
# ---------------------------------------------------------------------------

if not is_authed():
    render_login_signup()
    st.stop()


user = current_user()
assert user is not None  # is_authed() guarantees this


# ---------------------------------------------------------------------------
# Sidebar — nav + user card + manual refresh
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(f"### 🛰️ Drift Triage Co-Pilot")
    st.caption("Self-healing MLOps control plane")

    st.divider()

    pages = {
        "Overview": "📊",
        "Investigations": "🔍",
        "HIL Approvals": "✋",
        "Model Registry": "📦",
        "Drift Monitor": "📈",
        "Queue & DLQ": "⚙️",
    }
    page = st.radio(
        "Navigation",
        list(pages.keys()),
        format_func=lambda k: f"{pages[k]}  {k}",
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown(f"**{user['username']}**")
    st.caption(f"Role: `{user['role']}`")
    if st.button("Sign out", use_container_width=True):
        logout()
        st.rerun()

    st.divider()
    if st.button("🔄 Refresh data", use_container_width=True, help="Clear cached API responses"):
        st.cache_data.clear()
        st.rerun()
    auto_refresh = st.checkbox("Auto-refresh (10s)", value=False)


# ---------------------------------------------------------------------------
# Route to page
# ---------------------------------------------------------------------------

PAGES = {
    "Overview": overview_page.render,
    "Investigations": investigations_page.render,
    "HIL Approvals": approvals_page.render,
    "Model Registry": registry_page.render,
    "Drift Monitor": drift_page.render,
    "Queue & DLQ": queue_page.render,
}

PAGES[page]()


# ---------------------------------------------------------------------------
# Optional auto-refresh — implemented as a soft loop, opt-in only
# ---------------------------------------------------------------------------

if auto_refresh:
    import time

    time.sleep(10)
    st.cache_data.clear()
    st.rerun()
