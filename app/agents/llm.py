"""Single, role-based chat-model factory.

This is the **only** place in the codebase where chat model classes are
instantiated. Tests monkey-patch ``get_chat_model`` here; nodes import this
module (not the function) so the patch is observed at call time.

Pinned models
-------------

``PINNED_MODELS`` maps every agent role to a specific, dated model identifier.
A model bump is a code change reviewed in a PR, not a runtime config flip.
This is deliberate: silent model swaps are one of the easiest ways to ship a
regression that no test catches. See ``prompts/CHANGELOG.md`` for the
parallel discipline applied to prompts.
"""

from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel

from app.config import Settings, get_settings

AgentRole = Literal["supervisor", "triage", "action", "comms", "classifier"]


PINNED_MODELS: dict[AgentRole, str] = {
    "supervisor": "claude-haiku-4-5-20251001",
    "triage": "claude-haiku-4-5-20251001",
    "action": "claude-haiku-4-5-20251001",
    "comms": "claude-haiku-4-5-20251001",
    "classifier": "claude-haiku-4-5-20251001",
}


def get_chat_model(role: AgentRole, *, settings: Settings | None = None) -> BaseChatModel:
    """Return a chat model configured for the given agent role.

    Args:
        role: The agent role; selects the pinned model id from
            ``PINNED_MODELS``.
        settings: Optional override for ``get_settings()``. Tests rarely use
            this — they prefer to monkey-patch this whole function via
            ``app.testing.fixtures.patch_chat_model``.

    Returns:
        A LangChain ``BaseChatModel`` ready to invoke.

    Raises:
        ValueError: If the configured provider has no API key.
        KeyError: If the role is unknown.
    """
    settings = settings or get_settings()
    model_id = PINNED_MODELS[role]
    provider = settings.llm_provider

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required when LLM_PROVIDER=google")
        return ChatGoogleGenerativeAI(
            model=model_id,
            google_api_key=settings.google_api_key.get_secret_value(),
            temperature=0,
            max_output_tokens=2048,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        return ChatAnthropic(
            model=model_id,
            api_key=settings.anthropic_api_key.get_secret_value(),
            temperature=0,
            max_tokens=2048,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        return ChatOpenAI(
            model=model_id,
            api_key=settings.openai_api_key.get_secret_value(),
            temperature=0,
        )

    raise ValueError(f"Unknown LLM provider: {provider}")


__all__ = ["PINNED_MODELS", "AgentRole", "get_chat_model"]