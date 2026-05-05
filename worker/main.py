"""Queue consumer — processes triage tasks with DLQ and retry logic."""
from __future__ import annotations

import json
import os
import time

import redis

QUEUE = "triage_tasks"
DLQ = "triage_tasks_dlq"
MAX_RETRIES = 3

_r = redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))


def process(payload: dict) -> None:
    decision = payload.get("decision")
    model_id = payload.get("model_id")
    print(f"[worker] processing model={model_id} decision={decision}")
    # TODO: implement remediation actions (rollback, retrain, etc.)


def main() -> None:
    print("[worker] started, listening on", QUEUE)
    while True:
        raw = _r.blpop(QUEUE, timeout=5)
        if raw is None:
            continue
        _, data = raw
        payload = json.loads(data)
        retries = payload.get("_retries", 0)
        try:
            process(payload)
        except Exception as exc:  # noqa: BLE001
            print(f"[worker] error: {exc}")
            if retries < MAX_RETRIES:
                payload["_retries"] = retries + 1
                _r.rpush(QUEUE, json.dumps(payload))
            else:
                _r.rpush(DLQ, json.dumps(payload))


if __name__ == "__main__":
    main()
