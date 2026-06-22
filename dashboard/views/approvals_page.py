"""HIL inbox — pending approvals for Production-touching actions."""
from __future__ import annotations

import streamlit as st

import api_client
from auth import require_role


ACTION_ICON = {
    "rollback": "⏪",
    "retrain_shadow": "🔁",
    "promote": "🚀",
}


def render() -> None:
    st.markdown("## ✋ HIL Approvals")
    st.caption(
        "The graph pauses on a checkpoint until a reviewer approves or rejects the proposed action. "
        "Approvals fire only for Production-touching actions (`retrain_shadow`, `rollback`)."
    )

    try:
        pending = api_client.list_pending_approvals()
    except api_client.ApiError as e:
        st.error(f"Failed to load approvals: {e}")
        return

    if not pending:
        st.success("Inbox empty — no approvals waiting.")
        return

    st.info(f"**{len(pending)}** approval{'s' if len(pending) != 1 else ''} pending review.")

    for approval in pending:
        _render_card(approval)


def _render_card(approval: dict) -> None:
    aid = approval["id"]
    action = approval.get("proposed_action") or {}
    action_type = action.get("action_type", "unknown")
    icon = ACTION_ICON.get(action_type, "⚙️")

    with st.container(border=True):
        header = f"{icon} **{action_type}** — investigation `{approval['investigation_id'][:8]}`"
        st.markdown(header)
        st.caption(f"Approval ID `{aid[:8]}` · created `{approval.get('created_at', '—')}`")

        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown("**Prompt to reviewer:**")
            st.markdown(f"> {approval.get('prompt_to_reviewer', '_(no prompt)_')}")

            if action:
                st.markdown("**Proposed action plan:**")
                st.write(f"- Type: `{action.get('action_type', '—')}`")
                st.write(f"- Target version: `{action.get('target_version', '—')}`")
                if action.get("justification"):
                    st.markdown(f"- Justification: {action['justification']}")

        with col_right:
            with st.form(f"resp_{aid}"):
                feedback = st.text_area(
                    "Feedback (optional)",
                    placeholder="Why are you approving / rejecting?",
                    height=100,
                )
                col_a, col_r = st.columns(2)
                approve_clicked = col_a.form_submit_button(
                    "✅ Approve", use_container_width=True, type="primary"
                )
                reject_clicked = col_r.form_submit_button(
                    "❌ Reject", use_container_width=True
                )

                if approve_clicked or reject_clicked:
                    if not require_role("reviewer"):
                        return
                    approved = approve_clicked
                    try:
                        api_client.respond_to_approval(aid, approved=approved, feedback=feedback)
                        verb = "approved" if approved else "rejected"
                        st.success(f"Approval {verb}. The graph will resume.")
                        st.cache_data.clear()
                        st.rerun()
                    except api_client.ApiError as e:
                        st.error(f"Failed to respond: {e}")
