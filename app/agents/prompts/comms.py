"""Comms prompt — drafts a Slack-shaped notification.

VERSION: 0.2.0

Design notes:
- The comms agent produces a concise notification for the ML team. If a
  human review occurred, the reviewer's feedback is appended so the
  notification reflects the actual decision.
- Output is a single string (not structured) — it's a Slack message, not
  a data contract.
"""

COMMS_PROMPT = """\
Draft a concise Slack notification (≤ 3 sentences) for the ML team about the following drift event:

Model: {model_id}
Severity: {severity}
Proposed action: {action_description}

The notification should clearly state the model, the severity, and what action is being taken.
Do not use markdown headers or bullet lists — just plain sentences suitable for a Slack message.
"""