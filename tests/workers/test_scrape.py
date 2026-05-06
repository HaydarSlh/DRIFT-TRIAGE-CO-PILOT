"""Unit tests for ``scrape_url``.

We hit a stubbed httpx so the test runs in milliseconds and doesn't need
the network. The behaviors we drill:

- Happy path returns the right payload shape.
- 5xx + connect errors raise ``TransientToolError``.
- 4xx (other than 429) raises ``PermanentToolError``.
- 429 raises ``TransientToolError``.

The @task wrapper translates raw transient/permanent errors through
tenacity into a final ``ToolError`` (after retries). We use a broad
``ToolError`` catch in the assertions and inspect the message for the
underlying classification.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.core.errors import ToolError
from app.workers.tasks.scrape import scrape_url


def _patched_client(*, status_code: int = 200, text: str = "<html/>") -> Any:
    """Build a mock for ``httpx.AsyncClient`` context manager."""
    client = MagicMock()
    response = MagicMock(status_code=status_code, text=text)
    client.__aenter__.return_value.get = AsyncMock(return_value=response)
    return client


async def _call(payload: dict[str, Any], *, redis: Any) -> Any:
    """Call the @task wrapper directly with a synthetic ctx."""
    return await scrape_url({"job_id": "test", "redis": redis}, payload)


async def test_happy_path_returns_payload(
    settings_override: None, fake_redis: Any
) -> None:
    fake_client = _patched_client(status_code=200, text="hello world")
    with patch("httpx.AsyncClient", return_value=fake_client):
        result = await _call({"url": "https://x.test"}, redis=fake_redis)
    assert result["status_code"] == 200
    assert result["content"] == "hello world"
    assert result["bytes_fetched"] == len(b"hello world")


async def test_5xx_raises_transient(settings_override: None, fake_redis: Any) -> None:
    fake_client = _patched_client(status_code=503)
    with (
        patch("httpx.AsyncClient", return_value=fake_client),
        pytest.raises(ToolError) as exc_info,
    ):
        await _call({"url": "https://x.test"}, redis=fake_redis)
    assert "503" in str(exc_info.value) or "transient" in str(exc_info.value).lower()


async def test_429_raises_transient(settings_override: None, fake_redis: Any) -> None:
    fake_client = _patched_client(status_code=429)
    with (
        patch("httpx.AsyncClient", return_value=fake_client),
        pytest.raises(ToolError) as exc_info,
    ):
        await _call({"url": "https://x.test"}, redis=fake_redis)
    assert "429" in str(exc_info.value) or "transient" in str(exc_info.value).lower()


async def test_404_raises_permanent(settings_override: None, fake_redis: Any) -> None:
    fake_client = _patched_client(status_code=404)
    with (
        patch("httpx.AsyncClient", return_value=fake_client),
        pytest.raises(ToolError) as exc_info,
    ):
        await _call({"url": "https://x.test"}, redis=fake_redis)
    assert "404" in str(exc_info.value) or "permanent" in str(exc_info.value).lower()


async def test_invalid_url_raises_permanent_via_pydantic(
    settings_override: None, fake_redis: Any
) -> None:
    """Bad URL → Pydantic ValidationError, surfaced through the wrapper."""
    with pytest.raises(ToolError):
        await _call({"url": "not-a-url"}, redis=fake_redis)


async def test_connect_error_raises_transient(
    settings_override: None, fake_redis: Any
) -> None:
    """``httpx.ConnectError`` → TransientToolError → ToolError after retries."""
    fake_client = MagicMock()
    fake_client.__aenter__.return_value.get = AsyncMock(
        side_effect=httpx.ConnectError("boom")
    )
    with (
        patch("httpx.AsyncClient", return_value=fake_client),
        pytest.raises(ToolError) as exc_info,
    ):
        await _call({"url": "https://x.test"}, redis=fake_redis)
    assert (
        "connect" in str(exc_info.value).lower()
        or "transient" in str(exc_info.value).lower()
    )
