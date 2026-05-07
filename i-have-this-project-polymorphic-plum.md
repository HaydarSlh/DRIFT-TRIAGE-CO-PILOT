# Adapt `agenticPilot/` for Drift Triage Co-Pilot — Multi-Agent Side

## Context

`agenticPilot/` is a fully built multi-agent system from a prior bootcamp checkpoint (CP4 — "Long-Running Tools"). Its original domain was **research report generation** (supervisor → researcher → critic → writer with HIL review before finalizing the report). Architecturally it already satisfies almost every Week 5 requirement on the agent side:

- True LangGraph supervisor topology (supervisor is the only router; workers don't talk to each other)
- Postgres + Redis split: Postgres for the long-term audit trail, Redis for short-term LangGraph checkpoints
- arq Redis queue with deterministic idempotency keys, two-layer retries (tenacity inner + arq outer), and a dead-letter queue with admin endpoints
- Dynamic interrupt for human-in-the-loop approval (`writer_node` calls `interrupt(...)`, graph resumes via `Command(resume=...)`)
- Prompts as code (one `.py` file per role, version-tracked in `agents/prompts/CHANGELOG.md`)
- Test scaffolding: `FakeChatModel`, `TrajectoryRecorder`, syrupy snapshots, deterministic mode

The Week 5 brief asks for the **same architectural shape** but a different domain: drift events instead of topic submissions, triage/action/comms sub-agents instead of researcher/critic/writer, and remediation tools (replay test, retrain, rollback) instead of web-search/scrape/document-process.

The root-level `agent/` folder is a thin scaffold that names the right concepts (triage/action/comms, drift webhook, queue dispatch) but has no real LangGraph wiring, no checkpointer, no HIL, and no tests. We will mine it for prompt wording / schema names and then delete it.

**Goal of this plan:** remap `agenticPilot/` in place so its substrate (queue, checkpointer, services, repositories, tests, alembic) carries over unchanged, and only the *domain layer* (state shape, node logic, prompts, schemas, table semantics) is rewritten to fit drift triage.

**Decisions locked in (per user):**
- Folder name stays `agenticPilot/` — no top-level rename.
- Inner Python package name stays `app` — every existing `from app.config import ...` keeps working unchanged.
- HIL pause fires **only on Production-touching actions** (`retrain_shadow`, `rollback`); `none` and `replay_test` complete without pausing. This matches the brief's wording: "pause for human approval before any change to Production."

> **Scope:** multi-agent side only. The `platform/` (FastAPI + MLflow + drift compute) is the partner's track and is out of scope here.

---

## High-level mapping (research → drift triage)

| Concept in agenticPilot | Becomes in drift-triage |
|---|---|
| `ResearchState` (topic, findings, critique, report) | `TriageState` (drift_event, severity, recommended_action, draft_comms) |
| Node: **researcher** (gathers findings) | Node: **triage** (gathers context: drift report, recent runs, model card, perf metrics) |
| Node: **critic** (reviews findings) | Node: **action** (decides remediation: replay / retrain / rollback / no-op) |
| Node: **writer** (produces report + HIL interrupt) | Node: **comms** (drafts notification + HIL interrupt before any Production action) |
| Tool: `web_search` | Tool: `replay_test` (re-score the test set against current Production) |
| Tool: `scrape` | Tool: `retrain` (kick off a training run via the platform) |
| Tool: `document_process` | Tool: `rollback` (revert Production to previous registered version) |
| Table: `reports` | Table: `investigations` (one per drift webhook) |
| Table: `report_runs` | Table: `investigation_steps` (audit trail per node visit) |
| Table: `review_requests` | Table: `approval_requests` (HIL gate before any Production-touching action) |
| Webhook source: user POSTs `/reports` | Webhook source: platform POSTs `/webhooks/drift` |
| HIL approve → graph resumes → writer finalizes report | HIL approve → graph resumes → action tool calls platform's `/promote` endpoint |

---

## Folder-by-folder plan

Top-level rename: **`agenticPilot/` → `agent/`** (replacing the current draft `agent/`). The package import root inside the code is currently `app.*` (e.g. `from app.config import Settings`) — keep `app` as the Python package name to avoid touching every import.

> Decision to confirm with user: rename to `agent/` (matches docker-compose service name and the brief's "Triage Agent" box) **vs** keep `agenticPilot/` (no rename churn). I'll ask via AskUserQuestion before finalizing.

> Throughout the rest of this plan, paths are written as `agenticPilot/...` (the folder kept its name). The Python package name remains `app`, so all imports like `from app.agents.state import ...` keep working unchanged.

### `agenticPilot/agents/` — graph + nodes + prompts (heavy rewrite)

| File | Action | Purpose after edit |
|---|---|---|
| `state.py` | Rewrite | New `TriageState` TypedDict: `drift_event` (DriftEvent), `context` (RetrievedContext), `severity` (str), `recommended_action` (ActionPlan), `comms_draft` (str \| None), plus existing `thread_id`, `requires_human_review`, `human_feedback`, `messages`, `step_count`. Keep `add_messages` reducer. `Finding` becomes `RetrievedContext` (drift_report_id, recent_predictions_summary, model_card_ref, perf_window_metrics). |
| `graph.py` | Light edit | Same shape: 4 nodes (supervisor + 3 workers), START → supervisor, build_graph(checkpointer). Just rename node imports. |
| `nodes/supervisor.py` | Light edit | Same router pattern (`with_structured_output(SupervisorDecision)`, MAX_STEPS guardrail). Choices change to `triage | action | comms | END`. |
| `nodes/researcher.py` → `nodes/triage.py` | Rewrite | Gathers context: calls `platform_client.get_drift_report(model_id)`, `get_recent_predictions()`, `get_model_card()`. Outputs `RetrievedContext` + `severity` (LOW/MEDIUM/HIGH/CRITICAL). |
| `nodes/critic.py` → `nodes/action.py` | Rewrite | Pure LLM call: takes (drift_event, context, severity), produces `ActionPlan` (action_type ∈ {none, replay_test, retrain_shadow, rollback}, justification, target_version). Sets `requires_human_review=True` **only** when `action_type ∈ {retrain_shadow, rollback}` — Production-touching actions. `none` and `replay_test` proceed without pausing. |
| `nodes/writer.py` → `nodes/comms.py` | Rewrite | Drafts notification (Slack-shaped string for now). On entry, if `requires_human_review` is True → calls `interrupt({...})` with the proposed action plan as the prompt-to-reviewer. On resume, reads `human_feedback`. If approved, dispatches the action tool through the queue (`call_tool("rollback", payload, idempotency_key=thread_id+":rollback")`); if rejected, ends without action. |
| `nodes/_llm.py` | Keep as-is | Legacy back-compat shim — leave alone. |
| `llm.py` | Edit pinned-models map | Keys become `supervisor`, `triage`, `action`, `comms`, plus `classifier` (kept for any structured-output classifier reuse). |
| `checkpointer.py` | Keep as-is | Redis checkpointer build is domain-agnostic. |
| `tools/queued.py` | Keep as-is | `call_tool()` wrapper, fallback policy, deterministic id — domain-agnostic. |
| `prompts/researcher.py` → `prompts/triage.py` | Rewrite content | Prompt: "given drift event and PSI/chi² report, classify severity and summarize what changed." |
| `prompts/critic.py` → `prompts/action.py` | Rewrite content | Prompt: "propose remediation action; valid set = {none, replay_test, retrain_shadow, rollback}; justify in ≤3 sentences." Pin the legal action set in the schema, not the prompt body. |
| `prompts/writer.py` → `prompts/comms.py` | Rewrite content | Prompt: "draft a 3-sentence Slack message. If `human_feedback` is set, incorporate it." |
| `prompts/supervisor.py` | Light edit | Update node names in the routing prompt. |
| `prompts/CHANGELOG.md` | Append entry | "Forked from CP4 research-report; rewrote for Week 5 drift triage." |

### `agenticPilot/api/` — HTTP surface (moderate edit)

| File | Action | Purpose after edit |
|---|---|---|
| `health.py` | Keep as-is | Liveness/readiness/metrics endpoints are domain-agnostic. |
| `reports.py` → `investigations.py` | Rename + edit | Endpoints become: `GET /investigations`, `GET /investigations/{id}`, `POST /investigations/{id}/resume` (testing affordance). The *creation* path is no longer user-driven — it's the drift webhook. |
| `reviews.py` → `approvals.py` | Rename + edit | `GET /approvals/pending`, `POST /approvals/{id}/respond` — same flow, renamed. |
| `jobs.py` | Keep as-is | DLQ admin endpoints are domain-agnostic. |
| **NEW** `webhooks.py` | New file | `POST /webhooks/drift` — receives `DriftAlertEvent` from the platform, calls `InvestigationService.start_from_drift_event(...)`. The platform's webhook contract lives in `schemas/webhooks.py`. Verify a shared secret HMAC header (the brief calls out "treat schema changes as breaking" → version this contract). |

### `agenticPilot/db/` — ORM (table-name rename, schema mostly unchanged)

| Table now | Table after | Notes |
|---|---|---|
| `reports` | `investigations` | Add columns: `drift_event_id` (string, idempotency key for webhook redelivery), `model_id`, `model_version`, `severity`. Keep `thread_id`, `status`, timestamps. Lifecycle states unchanged in shape: `PENDING / RUNNING / AWAITING_APPROVAL / COMPLETED / FAILED`. |
| `report_runs` | `investigation_steps` | No schema change beyond the FK rename. |
| `review_requests` | `approval_requests` | Add column: `proposed_action` (JSONB) so the dashboard can render the HIL prompt without re-reading the checkpoint. |

`db/base.py` and `db/session.py` need no changes — async engine + sessionmaker is domain-neutral.

### `agenticPilot/services/` and `agenticPilot/repositories/` (rename + light edit)

- `report_service.py` → `investigation_service.py` — same astream/interrupt detection logic; rename calls.
- `review_service.py` → `approval_service.py` — same.
- `report_repo.py` → `investigation_repo.py`; `review_repo.py` → `approval_repo.py`.
- New method on `InvestigationService`: `start_from_drift_event(event)` — idempotent on `drift_event_id` (look up first; if found, return existing thread_id; this handles webhook redelivery cleanly).

### `agenticPilot/schemas/` (rewrite content; structure unchanged)

- `reports.py` → `investigations.py`: DTOs for the GET endpoints.
- `reviews.py` → `approvals.py`: DTOs for the approval endpoints.
- **NEW** `webhooks.py`: `DriftAlertEvent` (model_id, model_version, severity, drift_report_id, psi, chi2, output_drift, triggered_at, event_id). This is the shared contract with the platform — version it (`schema_version: Literal["1.0"]`) per the brief.

### `agenticPilot/workers/` and `agenticPilot/queue/` — async tools (rewrite tasks; keep substrate)

| File | Action | Purpose after edit |
|---|---|---|
| `queue/client.py` | Keep as-is | Producer (deterministic id, pre-flight check) is domain-neutral. |
| `queue/tasks.py` | Keep as-is | `@task` decorator, two-layer retry, idempotency cache. |
| `queue/results.py` | Keep as-is | JobResult storage. |
| `queue/dlq.py` | Keep as-is | DLQ + admin webhook. |
| `workers/runner.py` | Light edit | Update task imports to the new task modules. |
| `workers/tasks/web_search.py` | Replace with `replay_test.py` | Calls platform `POST /predict/replay` with the held-out test set; deterministic for tests. `retry_on=(TransientToolError,)`. |
| `workers/tasks/scrape.py` | Replace with `retrain.py` | Calls platform `POST /models/train`; returns new candidate version. Permanent failures (4xx, schema mismatch) → no retry. |
| `workers/tasks/document_process.py` | Replace with `rollback.py` | Calls platform `POST /registry/promote` with `target_version=previous`. **Critical**: idempotency key must encode the *target version*, not just the model_id, so two retries against different target versions don't collide. |

### `agenticPilot/core/` (no change)

`errors.py` (AgentError, ToolError, TransientToolError, PermanentToolError) and `logging.py` (structlog with request_id contextvar) carry over verbatim.

### `agenticPilot/testing/` (mostly keep; one rename)

- `fixtures.py`, `fakes.py` (`FakeChatModel`), `recorders.py` (`TrajectoryRecorder`) — keep as-is, all domain-neutral.
- New trajectory fixtures committed under `tests/agent/__snapshots__/` capturing the canonical drift-event → triage → action → HIL → comms paths (≥2: one no-op path, one Production-touching path). These are what CI's snapshot regression test pins.

### Top-level files

| File | Action |
|---|---|
| `agenticPilot/main.py` | Light edit: register `webhooks.router`, swap `reports`/`reviews` for `investigations`/`approvals`. Lifespan stays: build engine, sessionmaker, checkpointer, queue client, graph. |
| `agenticPilot/deps.py` | Rename: `get_report_service` → `get_investigation_service`; `get_review_service` → `get_approval_service`. Queue-client deps unchanged. |
| `agenticPilot/config.py` | Add: `platform_url`, `platform_webhook_secret` (HMAC), `dashboard_callback_url`. Keep all queue/checkpoint/LLM settings. |
| `agenticPilot/Dockerfile.worker` | No change — entry point is still `python -m app.workers.runner`. |
| `agenticPilot/langgraph.json` | Update entry-point comment; topology change is invisible to Studio (still 4 nodes through the supervisor). |
| `agenticPilot/pyproject.toml` | Rename project to `drift-triage-agent`; update console-script names; bump version to `0.1.0`. |
| `agenticPilot/INSTRUCTOR_NOTES.md` | Replace with `ARCH.md` content (the brief requires `ARCH.md`, `DECISIONS.md`, `RUNBOOK.md`). |

### Alembic — root vs nested (decision)

`alembic/` belongs at the **repo root**, not inside `agenticPilot/`. The Postgres instance is shared with the dashboard and (likely) the platform, and the brief explicitly calls out a single `docker-compose up`. Root layout:

```
DRIFT-TRIAGE-CO-PILOT/
├── alembic/
│   ├── env.py            ← imports app.db.models and (later) platform.db.models
│   └── versions/
│       └── 0001_initial_schema.py
├── alembic.ini
├── agenticPilot/         ← multi-agent service (folder name unchanged)
├── platform/
└── dashboard/
```

`env.py` imports the union of all services' SQLAlchemy `Base.metadata`, so a single `alembic upgrade head` migrates the whole stack. CI runs it before any service starts. (The IDE's open `agenticPilot/alembic/versions/...` tab is stale — the folder doesn't exist on disk yet; the migration would be created at the root location instead.)

### Discard

The root-level `agent/` folder (the draft scaffold from earlier today) is deleted once the `agenticPilot/` rewrite lands. Mine its `prompts/*.txt` files first — the wording is decent raw material for the new `agents/prompts/*.py` modules.

---

## What we keep verbatim (high-confidence reuse)

These files transfer with **zero or near-zero edits** because they're domain-neutral substrate:

- `agents/checkpointer.py` — Redis checkpointer build
- `agents/tools/queued.py` — producer-side `call_tool` with fallback policy
- `queue/client.py`, `queue/tasks.py`, `queue/results.py`, `queue/dlq.py` — full queue layer
- `core/errors.py`, `core/logging.py`
- `db/base.py`, `db/session.py`
- `testing/fakes.py`, `testing/recorders.py`, `testing/fixtures.py`
- `api/health.py`, `api/jobs.py`

---

## Verification plan

1. **`docker-compose up` from a clean clone**: agent service comes up, `/health/ready` returns ok, Postgres has the new tables (via `alembic upgrade head`).
2. **End-to-end happy path**: `POST /webhooks/drift` with a synthetic event → row appears in `investigations` (status RUNNING) → trajectory advances triage → action → comms → `approval_requests` row appears (status PENDING) → `POST /approvals/{id}/respond` with `approved=true` → queue dispatches `rollback` (or other) → JobResult lands → investigation completes.
3. **Idempotency**: post the same drift webhook twice with the same `event_id` → exactly one investigation row, exactly one queue job per action.
4. **Crash recovery**: kill the agent container mid-investigation (between `triage` and `action`); restart. Graph resumes from the last checkpoint, not from START. Verify by snapshot-comparing the run trajectory before/after kill.
5. **Snapshot trajectory tests** (`pytest -m snapshot`): with `FakeChatModel` patched in, the no-op path and the Production-touching path both match recorded fixtures.
6. **Kill-a-worker drill**: kill the worker mid-job; arq redelivers; `@task` idempotency check skips recomputation; JobResult is unchanged.
7. **CI**: GitHub Actions runs `alembic upgrade head` against ephemeral Postgres + Redis, then `pytest`. Refuse-merge on snapshot regression.

---

## Resolved questions

1. ✅ Top-level folder name: keep as `agenticPilot/`.
2. ✅ Inner Python package: keep as `app` (zero import churn).
3. ✅ HIL scope: pause only on Production-touching actions (`retrain_shadow`, `rollback`); `none` and `replay_test` complete without pausing.

---

## Additional task: lift dependency manifests to the repo root

Move `pyproject.toml` and `uv.lock` from `agenticPilot/` up to the repo root so they can be installed once for the whole stack (agent + platform + dashboard + worker share one Python env at the workspace level).

**Steps:**

1. **Read** `agenticPilot/pyproject.toml` to learn the existing dependency set, console scripts, and tool config (ruff, pytest, etc.).
2. **Read** the existing root `pyproject.toml` (the small placeholder created earlier today) to capture anything worth keeping.
3. **Write the merged `pyproject.toml` at the repo root** with:
   - Project name `drift-triage-co-pilot`, version `0.1.0`, Python ≥3.11.
   - Dependencies: union of agenticPilot's runtime deps (FastAPI, uvicorn, pydantic, langgraph, langchain-anthropic, langsmith, structlog, SQLAlchemy, asyncpg, psycopg2, alembic, redis, arq, tenacity, httpx) **plus** the platform-side deps (mlflow, scikit-learn, pandas, numpy, scipy) and dashboard deps (streamlit). Use optional-dependency groups (`[project.optional-dependencies]`) keyed by service: `agent`, `platform`, `worker`, `dashboard`, `dev` — so each Dockerfile can `pip install -e ".[agent]"` and pull only what it needs.
   - Console scripts re-pointed at `app.*` entry points (`triage-replay`, `triage-queue-admin`, `triage-worker`).
   - `[tool.ruff]`, `[tool.pytest.ini_options]` sections preserved from agenticPilot.
4. **Move `uv.lock`** from `agenticPilot/uv.lock` to the repo root verbatim. The lockfile pins the exact resolved versions agenticPilot was tested with; lifting it means all services build against the same locks. (If the union with platform's deps invalidates a few resolutions, run `uv lock` once at the root to regenerate; otherwise leave untouched.)
5. **Delete** the now-stale `agenticPilot/pyproject.toml` and `agenticPilot/uv.lock`.
6. **Update each Dockerfile** (`agenticPilot/Dockerfile.worker`, `platform/Dockerfile`, `agent/Dockerfile` once it's renamed/removed, `dashboard/Dockerfile`, `worker/Dockerfile`) to:
   - `COPY pyproject.toml uv.lock ./` from the build context (compose's build context is the repo root, not the per-service folder).
   - Run `uv sync --frozen --extra <service>` before copying the service code.
   - This makes Docker layer caching work properly: dependencies layer caches until `pyproject.toml` changes, source layer rebuilds on code change.
7. **Update `docker-compose.yml`** so each service's `build:` context points to the repo root with a per-service Dockerfile path:
   ```yaml
   agent:
     build:
       context: .
       dockerfile: agenticPilot/Dockerfile
   ```

**Why this matters:** the brief mandates `docker-compose up` from a clean clone. With per-service `pyproject.toml` files, each Docker build resolves its own subset of deps — slower, and risks version skew between the agent's `langgraph` and the platform's transitively-pinned `langgraph`. One root manifest = one resolution = one source of truth.

> Note: if the partner working on `platform/` already has their own `pyproject.toml`, merge carefully — don't clobber their dep pins. The dependency union step (3) is the place to negotiate that.
