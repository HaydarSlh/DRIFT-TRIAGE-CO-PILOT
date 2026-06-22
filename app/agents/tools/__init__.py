"""Agent-facing tool surface.

The triage and action agents (and any future tool-using node) import tools from here.
Two implementations behind one interface:

- ``app.agents.tools.queued``: enqueues a job, awaits the result.
- ``app.agents.tools.direct``: runs the worker function in-process.

Which one we use is driven by ``Settings.enable_queue``. When the queue is
on, the agent talks to the queue; when it is off, the agent runs the tool
inline. The agent code itself doesn't change.

Why two implementations and not "just always queue"? Two reasons:

1. Local dev. Students should be able to run the agent without booting a
   worker. Setting ``ENABLE_QUEUE=false`` is the escape hatch.
2. Tests. The unit tests for the agent nodes should not require a
   live arq worker. With the toggle set to false, the tests run the same
   tool body inline and assert on the agent's behavior.
"""

from app.agents.tools.queued import call_tool

__all__ = ["call_tool"]
