# Architecture — Drift Triage Co-Pilot

## Service Topology

```
Platform ──POST /webhooks/drift──▶ Agent (FastAPI + LangGraph)
                                      │
                                      ├── Supervisor (router)
                                      │     ├── Triage (gathers context)
                                      │     ├── Action  (decides remediation)
                                      │     └── Comms   (drafts notification + HIL)
                                      │
                                      ├── Postgres (investigations, steps, approval_requests)
                                      ├── Redis (LangGraph checkpoints + arq queue)
                                      └── Worker (arq: replay_test, retrain, rollback)
```

## Key Design Decisions

### 1. Supervisor topology (not peers)
The supervisor is the only router. Workers never call each other. This makes the graph observable and testable.

### 2. HIL only for Production-touching actions
`requires_human_review` is set to True only when the action is `retrain_shadow` or `rollback`. `none` and `replay_test` complete without human approval.

### 3. Postgres + Redis split
- **Postgres**: Long-term audit trail (investigations, investigation_steps, approval_requests)
- **Redis**: Short-term LangGraph checkpoints (per-step state, interrupt/resume)

### 4. Queue-backed tools
Actions that touch external services (replay test, retrain, rollback) go through the arq queue. The agent calls `call_tool()` which enqueues; the worker executes.

### 5. Idempotent webhook handling
`drift_event_id` is a unique key on the investigations table. Duplicate webhook redeliveries return 200 with `duplicate=True`.

### 6. Two-layer retries
Tenacity (inner) + arq (outer). TransientToolError triggers retry; PermanentToolError goes straight to the DLQ.

## Database Schema

- `investigations` — one row per drift webhook
- `investigation_steps` — audit trail per node visit
- `approval_requests` — HIL gate before Production-touching actions

## Folder Layout

```
agenticPilot/
├── agents/         — graph, nodes, state, prompts, tools
├── api/            — FastAPI routers (health, investigations, approvals, webhooks, jobs)
├── cli/            — replay, queue-admin CLIs
├── core/           — errors, logging (domain-neutral)
├── db/             — ORM models, session, base
├── queue/          — arq producer, results, DLQ
├── repositories/   — SQLAlchemy query layer
├── schemas/        — Pydantic request/response DTOs
├── services/       — business logic (investigation, approval)
├── testing/        — FakeChatModel, TrajectoryRecorder, fixtures
└── workers/        — arq task implementations
```