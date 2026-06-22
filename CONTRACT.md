# Service Contracts

Version: 1.1.0

This document defines the HTTP contracts between the **Platform** (model service) and the **Agent** (LangGraph supervisor), and between the **Agent** and the **Dashboard**. Schema changes are **breaking**. When a new version of a contract is introduced, the header `X-Contract-Version` MUST be bumped, and both services updated accordingly.

---

## 1. Drift Alert Webhook (Platform → Agent)

**Endpoint:** `POST /webhooks/drift`  
**Contract Version Header:** `X-Contract-Version: drift-alert-v1`

### Request Body (JSON)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `timestamp` | string (ISO 8601) | yes | When the alert was generated. |
| `event_id` | string (UUID) | yes | Unique id of this alert. **Used as the idempotency key for all downstream Redis queue jobs** — duplicate `event_id` values must be no-ops at every layer. |
| `severity` | string | yes | `green`, `yellow`, or `red`. |
| `previous_severity` | string | yes | Previous severity that was emitted. Only sent on severity transitions — not on every recomputation. |
| `current_window` | object | yes | Description of the evaluation window. |
| `current_window.start` | string (ISO 8601) | yes | Start of window. |
| `current_window.end` | string (ISO 8601) | yes | End of window. |
| `current_window.num_predictions` | integer | yes | Number of predictions in the window. |
| `drift_details` | object | yes | Drift metrics. |
| `drift_details.psi` | object | yes | PSI per numeric feature. Keys are feature names, values are floats. `< 0.10` stable, `< 0.25` moderate, `>= 0.25` significant. |
| `drift_details.chi2` | object | yes | Chi‑squared statistic per categorical feature (raw test statistic, not p‑value or converted index). Keys are feature names, values are floats. Larger values indicate greater distribution shift; the agent uses comparison against critical values from chi‑square distribution tables. |
| `drift_details.output_drift` | float | yes | Difference in positive‑prediction proportion (current − reference). Positive = model predicting more positives than at training time, negative = fewer. Magnitude matters more than sign for triage decisions. |

### Example

```json
{
  "timestamp": "2026-05-04T09:15:00Z",
  "event_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "severity": "red",
  "previous_severity": "yellow",
  "current_window": {
    "start": "2026-05-04T08:45:00Z",
    "end": "2026-05-04T09:15:00Z",
    "num_predictions": 512
  },
  "drift_details": {
    "psi": { "euribor3m": 0.31, "cons_price_idx": 0.09 },
    "chi2": { "job": 18.2, "marital": 3.5 },
    "output_drift": 0.13
  }
}
```

### Responses

| Status | Body | Meaning |
|--------|------|---------|
| `202 Accepted` | `{"investigation_id": "<uuid>"}` | Event accepted, investigation opened asynchronously. |
| `200 OK` | `{"investigation_id": "<uuid>", "duplicate": true}` | Duplicate `event_id` — already processing, no new investigation opened. |
| `400 Bad Request` | `{"detail": "..."}` | Invalid payload or missing/unrecognised `X-Contract-Version`. |

---

## 2. Promotion Request (Agent → Platform)

**Endpoint:** `POST /registry/promote`
**Contract Version Header:** `X-Contract-Version: promotion-v1`

### Request Body (JSON)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `action` | string | yes | `promote` or `rollback`. |
| `model_version` | string | yes | MLflow version number to promote (e.g. `"3"`). For rollback, the version to roll back **to** (not from). |
| `investigation_id` | string (UUID) | yes | The agent investigation that triggered this. The platform logs this for audit. |
| `approved_by` | string | yes | Identifier of the human operator who approved in the HIL inbox. |
| `approval_timestamp` | string (ISO 8601) | yes | When the human clicked approve. The platform rejects requests where `approval_timestamp` is older than 1 hour — stale approvals must not act on superseded investigations. |
| `reason` | string | yes | Short human-readable explanation, written by the comms sub-agent. |

### Example

```json
{
  "action": "promote",
  "model_version": "2",
  "investigation_id": "c3d4e5f6-...",
  "approved_by": "ops-jane",
  "approval_timestamp": "2026-05-04T09:20:00Z",
  "reason": "Drift red on euribor3m (PSI 0.31). Retrained model v2 outperforms v1 on replay test (AUC +0.03)."
}
```

### Responses

| Status | Body | Meaning |
|--------|------|---------|
| `200 OK` | `{"status": "promoted", "new_production_version": "2", "mlflow_run_id": "..."}` | Promotion checklist passed, version is now Production. |
| `409 Conflict` | `{"detail": "Promotion checklist failed: <criterion>"}` | Valid request but checklist rejected the candidate. Agent should surface this back to HIL. |
| `410 Gone` | `{"detail": "Approval expired"}` | `approval_timestamp` older than 1 hour. |
| `422 Unprocessable Entity` | `{"detail": "..."}` | Malformed request body. |

---

## 3. HIL Inbox (Agent → Dashboard)

The dashboard reads pending approvals from the agent and posts human decisions back. These endpoints live on the **Agent** service.

**Contract Version Header:** `X-Contract-Version: hil-v1`

---

### 3a. List pending HIL requests

**Endpoint:** `GET /hil/pending`

### Response — `200 OK`

```json
[
  {
    "hil_id": "<uuid>",
    "investigation_id": "<uuid>",
    "created_at": "2026-05-04T09:18:00Z",
    "action": "promote",
    "model_version": "2",
    "reason": "Retrained model v2 ready. Replay AUC 0.94 vs baseline 0.91.",
    "context": {
      "drift_summary": "euribor3m PSI 0.31 (red). output_drift +0.13.",
      "checklist_results": {
        "test_auc_above_threshold": true,
        "recall_above_0_75": true,
        "schema_matches_production": true,
        "no_critical_drift": false
      }
    }
  }
]
```

---

### 3b. Submit a HIL decision

**Endpoint:** `POST /hil/{hil_id}/decision`

### Request Body (JSON)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `decision` | string | yes | `approved` or `rejected`. |
| `decided_by` | string | yes | Operator identifier (free text, shown in audit log). |
| `note` | string | no | Optional operator note, stored in audit log. |

### Example

```json
{
  "decision": "approved",
  "decided_by": "ops-jane",
  "note": "Checked replay metrics — looks good."
}
```

### Responses

| Status | Body | Meaning |
|--------|------|---------|
| `200 OK` | `{"status": "approved", "investigation_id": "<uuid>"}` | Decision recorded, agent graph resumed from checkpoint. |
| `404 Not Found` | `{"detail": "HIL request not found or already resolved."}` | Already decided or wrong ID. |
| `410 Gone` | `{"detail": "Investigation superseded by a newer event."}` | A newer drift event has already acted — this approval is stale. |

---

## General rules

- All endpoints MUST validate the `X-Contract-Version` header. Missing or unrecognised value → `400 Bad Request`.
- Feature names in `drift_details.psi` and `drift_details.chi2`use underscores (cons_price_idx), not dots — dots break JSON key parsing in some clients.
- All timestamps are UTC ISO 8601 with `Z` suffix.
- UUIDs are lowercase hyphenated (`xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`).
