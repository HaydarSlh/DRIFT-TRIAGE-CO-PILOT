"""Investigation service — orchestrates persistence and graph execution.

Service responsibilities:

- Create the ``Investigation`` row upfront so the API can return 202 with an id
  before the graph has done any real work.
- Handle idempotent webhook redelivery: if an investigation with the same
  ``drift_event_id`` already exists, return the existing one without creating
  a duplicate.
- Drive the graph in a background task, capturing per-node updates as
  ``InvestigationStep`` rows for the audit trail.
- Detect human-approval interrupts and create the ``ApprovalRequest``; transition
  the investigation to ``awaiting_approval``.
- Resume an interrupted run when a reviewer responds, and finalize on completion.

All DB access goes through repositories. The service imports SQLAlchemy types
purely for session annotations, never for queries — that is the rule.
"""

import json
import time
import uuid
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.agents.state import ActionPlan, DriftEvent, RetrievedContext
from app.core.logging import get_logger
from app.db.models import Investigation, InvestigationStatus
from app.repositories.approval_repo import ApprovalRepository
from app.repositories.investigation_repo import InvestigationRepository
from app.schemas.webhooks import DriftAlertEvent

log = get_logger(__name__)

_NODE_NAMES = {"supervisor", "triage", "action", "comms"}


def _truncate(value: Any, limit: int = 800) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + "…"


def _summarize_update(update: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "severity" in update:
        out["severity"] = update["severity"]
    if "recommended_action" in update:
        action = update["recommended_action"]
        out["action_type"] = action.action_type if isinstance(action, ActionPlan) else str(action)
    if "comms_draft" in update:
        out["comms_draft_chars"] = len(update.get("comms_draft") or "")
    if "next_agent" in update:
        out["next_agent"] = update["next_agent"]
    if "step_count" in update:
        out["step_count"] = update["step_count"]
    if "requires_human_review" in update:
        out["requires_human_review"] = update["requires_human_review"]
    if not out:
        out["raw"] = _truncate(update)
    return out


class InvestigationService:
    """The application-level entry point for investigation runs."""

    def __init__(
        self,
        sessionmaker: async_sessionmaker,
        graph: CompiledStateGraph,
    ) -> None:
        self._sessionmaker = sessionmaker
        self._graph = graph

    async def get_investigation(self, investigation_id: uuid.UUID) -> Investigation | None:
        async with self._sessionmaker() as session:
            return await InvestigationRepository(session).get(investigation_id)

    async def list_investigations(self, limit: int = 50) -> list[Investigation]:
        async with self._sessionmaker() as session:
            return list(await InvestigationRepository(session).list(limit=limit))

    async def start_from_drift_event(
        self, event: DriftAlertEvent
    ) -> tuple[Investigation, bool]:
        """Create an investigation from a drift webhook event.

        Idempotent: if an investigation with the same ``event_id`` already
        exists, return it with ``duplicate=True``.

        Returns:
            (investigation, is_duplicate) tuple.
        """
        async with self._sessionmaker() as session:
            repo = InvestigationRepository(session)
            existing = await repo.get_by_drift_event_id(event.event_id)
            if existing is not None:
                await session.commit()
                return existing, True

            thread_id = uuid.uuid4().hex
            severity = event.severity
            investigation = await repo.create(
                drift_event_id=event.event_id,
                model_id=event.model_id,
                model_version=event.model_version,
                severity=severity,
                thread_id=thread_id,
            )
            await session.commit()
            await session.refresh(investigation)
            return investigation, False

    async def run_in_background(
        self,
        investigation_id: uuid.UUID,
        event: DriftAlertEvent,
        *,
        requires_human_review: bool | None = None,
    ) -> None:
        """Drive the graph end-to-end (or up to an interrupt) for an investigation."""
        investigation = await self.get_investigation(investigation_id)
        if investigation is None:
            log.error("run_in_background_missing_investigation", investigation_id=str(investigation_id))
            return

        await self._set_status(investigation_id, InvestigationStatus.RUNNING)
        config = {"configurable": {"thread_id": investigation.thread_id}}

        drift_event = DriftEvent(
            event_id=event.event_id,
            model_id=event.model_id,
            model_version=event.model_version,
            drift_report_id=event.drift_report_id,
            psi=event.psi,
            chi2=event.chi2,
            output_drift=event.output_drift,
            severity=event.severity,
        )
        initial: dict[str, Any] = {
            "drift_event": drift_event,
            "step_count": 0,
            "thread_id": investigation.thread_id,
            "requires_human_review": requires_human_review if requires_human_review is not None else False,
        }

        try:
            await self._drive_graph(investigation_id, initial, config)
        except Exception as exc:
            log.exception("run_in_background_failed", investigation_id=str(investigation_id), error=str(exc))
            await self._set_status(investigation_id, InvestigationStatus.FAILED)
            return

        await self._finalize_after_run(investigation_id, investigation.thread_id, config)

    async def resume_investigation(
        self,
        thread_id: str,
        decision: dict[str, Any],
    ) -> None:
        """Resume an interrupted run with the reviewer's decision."""
        async with self._sessionmaker() as session:
            repo = InvestigationRepository(session)
            investigation = await repo.get_by_thread(thread_id)
            if investigation is None:
                raise LookupError(f"no investigation for thread_id={thread_id}")
            investigation_id = investigation.id

        config = {"configurable": {"thread_id": thread_id}}
        await self._set_status(investigation_id, InvestigationStatus.RUNNING)
        try:
            await self._drive_graph(investigation_id, Command(resume=decision), config)
        except Exception as exc:
            log.exception("resume_failed", thread_id=thread_id, error=str(exc))
            await self._set_status(investigation_id, InvestigationStatus.FAILED)
            return

        await self._finalize_after_run(investigation_id, thread_id, config)

    async def get_final_state(self, thread_id: str) -> dict[str, Any]:
        """Pull the final state out of the checkpointer for the GET endpoint."""
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = await self._graph.aget_state(config)
        values = snapshot.values or {}
        result: dict[str, Any] = {}
        context = values.get("context")
        if context:
            result["context"] = context if isinstance(context, RetrievedContext) else RetrievedContext.model_validate(context)
        severity = values.get("severity")
        if severity:
            result["severity"] = severity
        action = values.get("recommended_action")
        if action:
            result["recommended_action"] = action if isinstance(action, ActionPlan) else ActionPlan.model_validate(action)
        comms_draft = values.get("comms_draft")
        if comms_draft:
            result["comms_draft"] = comms_draft
        return result

    async def _drive_graph(
        self,
        investigation_id: uuid.UUID,
        graph_input: Any,
        config: dict[str, Any],
    ) -> None:
        existing = await self._count_existing_steps(investigation_id)
        step = existing
        async for chunk in self._graph.astream(
            graph_input, config=config, stream_mode="updates"
        ):
            for node_name, update in chunk.items():
                if node_name not in _NODE_NAMES:
                    continue
                step += 1
                started = time.perf_counter()
                duration_ms = int((time.perf_counter() - started) * 1000)
                await self._append_step(
                    investigation_id=investigation_id,
                    step_number=step,
                    agent_name=node_name,
                    output_summary=_summarize_update(update or {}),
                    duration_ms=duration_ms,
                )

    async def _finalize_after_run(
        self,
        investigation_id: uuid.UUID,
        thread_id: str,
        config: dict[str, Any],
    ) -> None:
        snapshot = await self._graph.aget_state(config)
        pending_interrupts = [
            i for task in (snapshot.tasks or ()) for i in (task.interrupts or ())
        ]

        if pending_interrupts:
            payload = pending_interrupts[0].value
            log.info(
                "run_paused_for_approval",
                investigation_id=str(investigation_id),
                thread_id=thread_id,
            )
            async with self._sessionmaker() as session:
                inv_repo = InvestigationRepository(session)
                appr_repo = ApprovalRepository(session)
                await inv_repo.update_status(investigation_id, InvestigationStatus.AWAITING_APPROVAL)
                prompt_text = (
                    payload.get("prompt", json.dumps(payload))
                    if isinstance(payload, dict)
                    else str(payload)
                )
                proposed_action = payload.get("action_plan") if isinstance(payload, dict) else None
                await appr_repo.create(
                    investigation_id=investigation_id,
                    prompt_to_reviewer=prompt_text,
                    proposed_action=proposed_action,
                )
                await session.commit()
            return

        async with self._sessionmaker() as session:
            repo = InvestigationRepository(session)
            await repo.update_status(investigation_id, InvestigationStatus.COMPLETED)
            await session.commit()
        values = snapshot.values or {}
        log.info(
            "run_completed",
            investigation_id=str(investigation_id),
            steps=values.get("step_count"),
        )

    async def _set_status(self, investigation_id: uuid.UUID, status: InvestigationStatus) -> None:
        async with self._sessionmaker() as session:
            repo = InvestigationRepository(session)
            await repo.update_status(investigation_id, status)
            await session.commit()

    async def _append_step(
        self,
        *,
        investigation_id: uuid.UUID,
        step_number: int,
        agent_name: str,
        output_summary: dict[str, Any],
        duration_ms: int,
    ) -> None:
        async with self._sessionmaker() as session:
            repo = InvestigationRepository(session)
            await repo.append_step(
                investigation_id=investigation_id,
                step_number=step_number,
                agent_name=agent_name,
                input_summary={},
                output_summary=output_summary,
                duration_ms=duration_ms,
            )
            await session.commit()

    async def _count_existing_steps(self, investigation_id: uuid.UUID) -> int:
        async with self._sessionmaker() as session:
            repo = InvestigationRepository(session)
            steps = await repo.list_steps(investigation_id)
            return len(steps)