"""Worker tasks.

Each module here defines exactly one task and registers it via the
``@task`` decorator. The decorator pushes the task into
``app.taskqueue.tasks.TASK_REGISTRY``; the worker runner picks them all up
from there.

Conventions:
- Input/output are JSON-serializable dicts. Pydantic models inside the
  function body are fine; the queue boundary is dicts.
- Tasks are idempotent. The decorator enforces that for retries via the
  job-result check, but the task body itself must also tolerate "this
  same input may have been partially processed before."
- Slow on purpose. Tasks model real long-running tools (HTTP calls to the
  platform API, LLM call, replay test). The point is the queue, not the speed.
"""
