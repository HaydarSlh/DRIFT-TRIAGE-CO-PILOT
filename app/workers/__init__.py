"""Worker process package.

The arq worker entry point lives in ``app.workers.runner``. The actual task
implementations live under ``app.workers.tasks`` — one module per task so
each can carry its own input/output schema and unit test.

Run:
    uv run python -m app.workers.runner

Or via the script alias:
    uv run triage-worker

Workers are stateless: any state they need they read from Redis (queue,
results, DLQ) or Postgres. Killing a worker mid-job is *the* test we care
about — see ``tests/integration/test_worker_crash.py``.
"""

# Importing the task modules at package init triggers their @task decorators,
# which populates ``app.taskqueue.tasks.TASK_REGISTRY``. The runner then reads the
# registry to wire up arq's ``functions`` list. Side-effect imports are
# unusual but the alternative (a manual registration call) is worse here:
# it means the runner has to remember to import every new task file.
from app.workers.tasks import (  # noqa: F401
    replay_test,
    retrain,
    rollback,
)
