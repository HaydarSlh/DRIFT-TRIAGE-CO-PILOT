"""Queue layer.

Public surface:

- ``QueueClient``: enqueue jobs, await results, check status, cancel.
- ``JobResult``, ``JobStatus``: terminal-state record returned by
  ``await_result``.
- ``task``: decorator to register a worker function with idempotency, retry,
  result-persistence, and DLQ-on-permanent-failure baked in.
- ``TASK_REGISTRY``: name → callable map consumed by the arq worker.

The lesson is the *separation*: agents never call long tools synchronously.
They enqueue a job and wait on its result. Whether the worker succeeds, fails
transiently, fails permanently, or crashes mid-execution is a queue concern,
not an agent concern.
"""

from app.taskqueue.client import QueueClient
from app.taskqueue.dlq import (
    list_dlq,
    purge_dlq,
    push_to_dlq,
    requeue_from_dlq,
)
from app.taskqueue.results import JobResult, JobStatus, load_result, store_result
from app.taskqueue.tasks import TASK_REGISTRY, task

__all__ = [
    "TASK_REGISTRY",
    "JobResult",
    "JobStatus",
    "QueueClient",
    "list_dlq",
    "load_result",
    "purge_dlq",
    "push_to_dlq",
    "requeue_from_dlq",
    "store_result",
    "task",
]
