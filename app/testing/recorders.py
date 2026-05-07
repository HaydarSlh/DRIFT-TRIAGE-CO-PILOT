"""Trajectory and state recorders.

``TrajectoryRecorder`` watches the lightweight node-level events emitted by
``graph.astream(stream_mode="updates")`` and produces a list of
``TrajectoryStep`` records suitable for snapshot tests.

``StateRecorder`` keeps a fuller history — one full state dict per node visit
— for time-travel debugging in the replay CLI.

Neither recorder requires pytest; they're plain classes the testing fixtures
glue onto a graph.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Trajectory recorder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrajectoryStep:
    """One step in a recorded trajectory.

    Stable, comparable, and JSON-serializable so syrupy can snapshot it.
    """

    step_number: int
    agent: str
    output_summary: dict[str, Any]


@dataclass
class TrajectoryRecorder:
    """Streams a graph and captures one TrajectoryStep per node visit.

    Snapshot tests assert ``recorder.trajectory == loaded_snapshot``. The
    ``output_summary`` is intentionally lossy (counts and types, not raw
    content) so prompt edits that produce different prose don't churn
    snapshots.
    """

    steps: list[TrajectoryStep] = field(default_factory=list)

    @property
    def trajectory(self) -> list[dict[str, Any]]:
        """List form for serialization. Use this in snapshot assertions."""
        return [
            {
                "step_number": s.step_number,
                "agent": s.agent,
                "output_summary": s.output_summary,
            }
            for s in self.steps
        ]

    @property
    def visited_agents(self) -> list[str]:
        return [s.agent for s in self.steps]

    async def run(
        self,
        graph: CompiledStateGraph,
        initial_state: dict[str, Any],
        *,
        config: dict[str, Any] | None = None,
    ) -> AsyncIterator[TrajectoryStep]:
        """Drive ``graph`` and yield each ``TrajectoryStep`` as it happens."""
        step = 0
        async for chunk in graph.astream(
            initial_state, config=config, stream_mode="updates"
        ):
            for node_name, update in chunk.items():
                step += 1
                summary = _summarize(update or {})
                trajectory_step = TrajectoryStep(
                    step_number=step, agent=node_name, output_summary=summary
                )
                self.steps.append(trajectory_step)
                yield trajectory_step


def _summarize(update: dict[str, Any]) -> dict[str, Any]:
    """Reduce a node's state-update to JSON-friendly summary fields."""
    out: dict[str, Any] = {}
    for key, value in update.items():
        if key == "findings":
            out["findings_count"] = len(value or [])
        elif key in {"critique", "report"}:
            out[f"{key}_chars"] = len(value or "")
        elif key in {"step_count", "next_agent"}:
            out[key] = value
        elif isinstance(value, list | dict | str | int | float | bool) or value is None:
            out[key] = _normalize(value)
        else:
            out[key] = type(value).__name__
    return out


def _normalize(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, Document):
        return {"page_content": value.page_content, "metadata": value.metadata}
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items()}
    return value


# ---------------------------------------------------------------------------
# State recorder (heavier — full state history)
# ---------------------------------------------------------------------------


@dataclass
class StateRecorder:
    """Captures the full state dict at every node end.

    Heavier than ``TrajectoryRecorder``; not for snapshot tests. The replay
    CLIs use it to walk a recorded run step by step.
    """

    snapshots: list[dict[str, Any]] = field(default_factory=list)

    async def run(
        self,
        graph: CompiledStateGraph,
        initial_state: dict[str, Any],
        *,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        async for chunk in graph.astream(
            initial_state, config=config, stream_mode="values"
        ):
            self.snapshots.append(_normalize(chunk))
        return self.snapshots[-1] if self.snapshots else {}
