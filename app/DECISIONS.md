# Decisions

## Locked In

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Folder name stays `agenticPilot/` | No top-level rename churn; inner package is already `app` |
| 2 | Python package name stays `app` | Every `from app.config import ...` keeps working |
| 3 | HIL fires only on Production-touching actions | `retrain_shadow` and `rollback` pause; `none` and `replay_test` complete without pause |
| 4 | Postgres for audit trail, Redis for checkpoints | Concern separation; short-term state expires naturally |
| 5 | arq for tool queue | Async-native, Redis-only, fits FastAPI's event loop |
| 6 | Idempotency via `drift_event_id` unique key | Handles webhook redelivery cleanly |
| 7 | Contract versioning with `X-Contract-Version` header | Platform and agent evolve independently |

## Open

| # | Question | Status |
|---|----------|--------|
| 1 | Whether to add MLflow model registry integration to the rollback tool | Deferred to platform team |