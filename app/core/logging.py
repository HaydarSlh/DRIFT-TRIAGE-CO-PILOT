"""Structured logging setup.

JSON output in non-local environments; human-friendly console output for local dev.
A request-id ContextVar is included in every log line so a single run can be traced
across nodes, services, and middleware without passing the id around explicitly.
"""

import logging
import sys
from contextvars import ContextVar
from typing import Any

import structlog

request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)


def _add_request_id(_logger: Any, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """structlog processor that surfaces the current request_id ContextVar."""
    rid = request_id_ctx.get()
    if rid:
        event_dict["request_id"] = rid
    return event_dict


def configure_logging(level: str = "INFO", env: str = "local") -> None:
    """Configure structlog and the stdlib logger.

    Args:
        level: Log level name (e.g. "INFO", "DEBUG").
        env: Application environment. "local" gets pretty console output;
            anything else gets JSON.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=log_level)

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.format_exc_info,
        _add_request_id,
    ]
    if env == "local":
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog-bound logger.

    Args:
        name: Logger name; usually ``__name__`` of the caller.
    """
    return structlog.get_logger(name) if name else structlog.get_logger()
