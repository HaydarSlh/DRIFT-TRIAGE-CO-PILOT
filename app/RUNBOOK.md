# Runbook — Drift Triage Co-Pilot

## Quick Start

```bash
# 1. Copy env and start
cp .env.example .env
docker-compose up --build

# 2. Run migrations
docker-compose exec agent alembic upgrade head

# 3. Verify health
curl http://localhost:8001/health/ready
```

## Sending a Drift Event

```bash
curl -X POST http://localhost:8001/webhooks/drift \
  -H "Content-Type: application/json" \
  -H "X-Contract-Version: drift-alert-v1" \
  -d '{
    "schema_version": "1.0",
    "timestamp": "2026-05-07T09:15:00Z",
    "event_id": "test-001",
    "model_id": "bank-marketing-v1",
    "model_version": "3",
    "severity": "red",
    "psi": {"euribor3m": 0.31},
    "chi2": {"job": 18.2},
    "output_drift": 0.13
  }'
```

## Checking Investigations

```bash
curl http://localhost:8001/investigations
curl http://localhost:8001/investigations/{id}
```

## Approving / Rejecting Actions

```bash
# List pending approvals
curl http://localhost:8001/approvals/pending

# Approve
curl -X POST http://localhost:8001/approvals/{id}/respond \
  -H "Content-Type: application/json" \
  -d '{"approved": true, "feedback": "Looks good"}'

# Reject
curl -X POST http://localhost:8001/approvals/{id}/respond \
  -H "Content-Type: application/json" \
  -d '{"approved": false, "feedback": "Need more data"}'
```

## Monitoring

```bash
# Health
curl http://localhost:8001/health
curl http://localhost:8001/health/ready
curl http://localhost:8001/health/llm

# Metrics (DLQ depth, worker pool)
curl http://localhost:8001/health/metrics

# DLQ admin
curl http://localhost:8001/jobs/dlq/list
curl -X POST http://localhost:8001/jobs/dlq/{job_id}/requeue
curl -X DELETE http://localhost:8001/jobs/dlq/{job_id}
```

## Crash Recovery

If the agent container dies mid-investigation:
1. Restart: `docker-compose up agent`
2. The graph resumes from the last Redis checkpoint — no data loss
3. Check investigation status: `curl http://localhost:8001/investigations/{id}`

## Worker Management

```bash
# Check worker health
curl http://localhost:8001/health/ready

# Queue admin CLI
uv run triage-queue-admin status
uv run triage-queue-admin dlq list
```