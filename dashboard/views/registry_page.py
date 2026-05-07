"""Model Registry — MLflow versions, stages, metrics, manual promote/rollback."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

import api_client
from auth import current_user, require_role


STAGE_COLOR = {
    "Production": "🟢",
    "Staging": "🟡",
    "Archived": "⚫",
    "None": "⚪",
}


def render() -> None:
    st.markdown("## 📦 Model Registry")
    st.caption(
        "Live view of MLflow's registered models. The `promote` action runs the "
        "platform's day-4 promotion checklist before any stage change."
    )

    models = api_client.list_registered_models()
    if not models:
        st.warning(
            "No registered models found. Either MLflow is unreachable or the platform "
            "hasn't trained the first model yet."
        )
        return

    names = [m.get("name") for m in models]
    chosen_name = st.selectbox("Registered model", names)

    if not chosen_name:
        return

    versions = api_client.list_model_versions(chosen_name)
    if not versions:
        st.info(f"No versions registered for `{chosen_name}` yet.")
        return

    df = pd.DataFrame(versions)

    # Surface metrics from each linked run (best-effort; cached)
    metric_rows = []
    for v in versions:
        run_id = v.get("run_id", "")
        run = api_client.get_run(run_id) if run_id else {}
        metrics = (run.get("run", {}).get("data", {}).get("metrics") or []) if run else []
        m_dict = {m["key"]: m["value"] for m in metrics} if metrics else {}
        metric_rows.append(
            {
                "version": v.get("version"),
                "stage": v.get("current_stage", "None"),
                "run_id": run_id[:8] if run_id else "—",
                "val_auc": round(m_dict.get("val_auc", float("nan")), 4),
                "val_recall": round(m_dict.get("val_recall", float("nan")), 4),
                "operating_threshold": round(m_dict.get("operating_threshold", float("nan")), 4),
                "creation_timestamp": _ts(v.get("creation_timestamp")),
            }
        )

    metrics_df = pd.DataFrame(metric_rows).sort_values("version", ascending=False)
    metrics_df["stage"] = metrics_df["stage"].map(lambda s: f"{STAGE_COLOR.get(s, '•')} {s}")

    st.markdown("### Versions")
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # Highlight current production version
    prod_versions = [v for v in versions if v.get("current_stage") == "Production"]
    if prod_versions:
        prod = prod_versions[0]
        st.success(f"🟢 **Production:** version `{prod.get('version')}` (run `{prod.get('run_id', '')[:8]}`)")
    else:
        st.warning("No version currently in **Production**.")

    st.divider()

    # ------------------------------------------------------------------
    # Promote / Rollback (admin only)
    # ------------------------------------------------------------------
    st.markdown("### Manual stage transition")
    st.caption(
        "Bypasses the agent. The brief asks: _can the platform's promotion endpoint be called "
        "without going through the agent — and should it?_ Use this only for dry-runs or "
        "emergency interventions; everyday promotions should go through the agent + HIL flow."
    )

    if not current_user() or current_user().get("role") != "admin":
        st.info("Only **admin** users can trigger stage transitions from the dashboard.")
        return

    col1, col2 = st.columns(2)
    target_version = col1.selectbox(
        "Target version",
        [v.get("version") for v in versions],
        help="Which registered version to transition.",
    )
    action = col2.selectbox("Action", ["promote", "rollback"])

    if st.button(f"Execute {action} → v{target_version}", type="primary"):
        if not require_role("admin"):
            return
        try:
            now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            res = api_client.promote_version(
                model_version=str(target_version),
                action=action,
                approved_by=current_user()["username"],
                approval_ts=now_iso,
            )
            st.success(f"Done: {res}")
            st.cache_data.clear()
        except api_client.ApiError as e:
            st.error(f"Promotion failed: {e}")


def _ts(ms: int | None) -> str:
    if not ms:
        return "—"
    try:
        return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "—"
