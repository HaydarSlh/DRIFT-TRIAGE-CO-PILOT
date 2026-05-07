"""Domain exception types and the FastAPI handlers that map them to HTTP responses."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.logging import get_logger

log = get_logger(__name__)


class AgentError(Exception):
    """Raised when an agent node fails to produce a usable result."""


class ValidationError(Exception):
    """Raised when an inbound payload is structurally valid but semantically wrong.

    Distinct from Pydantic's ValidationError, which is raised for shape mismatches
    before our handlers ever see the request.
    """


class ProviderError(Exception):
    """Raised when an upstream LLM provider returns an error."""


# ---------------------------------------------------------------------------
# Tool / queue layer (CP4)
# ---------------------------------------------------------------------------


class ToolError(Exception):
    """Base class for queue-backed tool failures.

    Distinct from ``AgentError`` so we can map tool errors specifically (the
    agent saw the queue but the *tool* failed) without polluting the broader
    agent error path.
    """


class TransientToolError(ToolError):
    """Retryable. Network blip, 5xx, rate-limit."""


class PermanentToolError(ToolError):
    """Non-retryable. 4xx, schema mismatch, "you'll never succeed."""


def install_exception_handlers(app: FastAPI) -> None:
    """Register JSON exception handlers for the application's domain errors.

    Args:
        app: The FastAPI application to attach handlers to.
    """

    @app.exception_handler(AgentError)
    async def _agent_error(_req: Request, exc: AgentError) -> JSONResponse:
        log.warning("agent_error", error=str(exc))
        return JSONResponse(
            status_code=502,
            content={"error": "agent_error", "detail": str(exc)},
        )

    @app.exception_handler(ValidationError)
    async def _validation_error(_req: Request, exc: ValidationError) -> JSONResponse:
        log.warning("validation_error", error=str(exc))
        return JSONResponse(
            status_code=400,
            content={"error": "validation_error", "detail": str(exc)},
        )

    @app.exception_handler(ProviderError)
    async def _provider_error(_req: Request, exc: ProviderError) -> JSONResponse:
        log.error("provider_error", error=str(exc))
        return JSONResponse(
            status_code=502,
            content={"error": "provider_error", "detail": str(exc)},
        )
