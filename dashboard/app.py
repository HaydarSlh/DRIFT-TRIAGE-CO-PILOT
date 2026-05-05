"""Streamlit dashboard — human-in-the-loop (HIL) review interface."""
import os

import httpx
import streamlit as st

PLATFORM_URL = os.environ.get("PLATFORM_URL", "http://localhost:8000")

st.set_page_config(page_title="Drift Triage Co-Pilot", layout="wide")
st.title("Drift Triage Co-Pilot")

st.header("Active Drift Alerts")

try:
    reports = httpx.get(f"{PLATFORM_URL}/drift/", timeout=5).json()
except Exception:
    reports = []
    st.warning("Could not reach platform service.")

if not reports:
    st.info("No active drift alerts.")
else:
    for report in reports:
        with st.expander(f"Model: {report.get('model_id')} — {report.get('decision', 'pending')}"):
            st.json(report)
            col1, col2 = st.columns(2)
            if col1.button("Approve", key=f"approve_{report.get('id')}"):
                st.success("Approved")
            if col2.button("Reject", key=f"reject_{report.get('id')}"):
                st.error("Rejected")
