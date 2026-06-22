"""Compatibility shim — superseded by ``app.agents.llm.get_chat_model``.

CP3 centralizes chat-model instantiation in ``app.agents.llm``. This file
exists so the existing CP1/CP2 monkey-patch idiom
(``monkeypatch.setattr('app.agents.nodes._llm.get_llm', ...)``) keeps working
on legacy tests during the migration. New tests should patch
``app.agents.llm.get_chat_model`` instead — see ``app.testing.fixtures``.
"""

from typing import Any

from app.agents import llm as llm_module
from app.config import Settings


def get_llm(settings: Settings, *, role: llm_module.AgentRole = "triage") -> Any:
    """Back-compat wrapper for the CP1-era ``get_llm(settings)`` signature."""
    return llm_module.get_chat_model(role, settings=settings)