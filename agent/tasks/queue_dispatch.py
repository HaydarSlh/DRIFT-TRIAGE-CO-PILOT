"""Dispatch remediation tasks onto the Redis queue."""
import json
import os
import uuid

import redis

_r = redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
QUEUE = "triage_tasks"


def dispatch(model_id: str, decision: str) -> str:
    task_id = str(uuid.uuid4())
    payload = {"task_id": task_id, "model_id": model_id, "decision": decision}
    _r.rpush(QUEUE, json.dumps(payload))
    return task_id
