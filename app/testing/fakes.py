"""Deterministic stand-ins for the LLM / embeddings / retriever clients.

These are the **only** mocks the test pyramid leans on. The contract:

1. Constructed with an iterable of canned responses (or a schema-keyed map for
   structured-output tests).
2. ``ainvoke`` (or ``invoke``) returns the next response in order — or, with
   ``match_substring=True``, returns the first canned response whose key
   appears as a substring of the prompt.
3. ``call_history`` is populated for assertions: list of
   ``(messages_repr, schema_or_None)`` tuples, in invocation order.
4. Calling more times than canned responses is a hard error — fail loud, not
   silent.

Tests that need richer behaviour build a small subclass; these fakes are
deliberately small and obvious.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Chat model
# ---------------------------------------------------------------------------


@dataclass
class _CallRecord:
    """One recorded invocation. Used by tests for assertions."""

    messages_repr: str
    schema: type | None
    kwargs: dict[str, Any]


class _Plain:
    """Minimal AIMessage-shaped object — just enough for nodes that read
    ``response.content``."""

    def __init__(self, content: str) -> None:
        self.content = content


class _StructuredFakeRunnable:
    """Runnable returned by ``FakeChatModel.with_structured_output``.

    Held separately so the parent's plain ``ainvoke`` doesn't accidentally
    consume schema state set up for a different node.
    """

    def __init__(self, schema: type, parent: "FakeChatModel") -> None:
        self._schema = schema
        self._parent = parent

    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        return await self._parent._dispatch(
            messages=messages, schema=self._schema, kwargs=kwargs
        )

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        # Sync fallback — convert the canned response synchronously.
        # We don't use this in tests but the contract accepts it.
        import asyncio

        return asyncio.run(self.ainvoke(messages, **kwargs))


class FakeChatModel:
    """Deterministic stand-in for ``ChatAnthropic`` / ``ChatOpenAI``.

    Construction modes:

    - ``FakeChatModel(responses=[...])`` — return canned responses in order.
    - ``FakeChatModel(by_substring={"foo": resp1, "bar": resp2})`` — match the
      first key that appears in the prompt's ``repr``. Useful when the test
      doesn't want to depend on exact node ordering.
    - ``FakeChatModel(by_schema={SchemaA: respA, SchemaB: respB})`` — return
      ``respA`` whenever ``with_structured_output(SchemaA)`` is bound. Most
      common pattern for testing structured-output nodes.

    Tests can mix and match: ``by_schema`` takes precedence over
    ``by_substring`` over the ordered list.
    """

    def __init__(
        self,
        *,
        responses: Iterable[Any] | None = None,
        by_substring: dict[str, Any] | None = None,
        by_schema: dict[type, Any] | None = None,
    ) -> None:
        self._responses: list[Any] = list(responses or [])
        self._by_substring = dict(by_substring or {})
        self._by_schema = dict(by_schema or {})
        self.call_history: list[_CallRecord] = []

    def with_structured_output(self, schema: type) -> _StructuredFakeRunnable:
        return _StructuredFakeRunnable(schema, self)

    def bind(self, **_kwargs: Any) -> "FakeChatModel":
        return self

    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        return await self._dispatch(messages=messages, schema=None, kwargs=kwargs)

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        import asyncio

        return asyncio.run(self.ainvoke(messages, **kwargs))

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    async def _dispatch(
        self, *, messages: Any, schema: type | None, kwargs: dict[str, Any]
    ) -> Any:
        self.call_history.append(
            _CallRecord(messages_repr=_render(messages), schema=schema, kwargs=dict(kwargs))
        )
        # 1. schema-specific
        if schema is not None and schema in self._by_schema:
            return self._by_schema[schema]
        # 2. substring-keyed
        if self._by_substring:
            blob = _render(messages)
            for key, value in self._by_substring.items():
                if key in blob:
                    return value
        # 3. ordered list — fall back to a plain text response if no schema set
        if self._responses:
            value = self._responses.pop(0)
            return value if schema is not None else _coerce_to_message(value)
        raise AssertionError(
            f"FakeChatModel exhausted (schema={schema}, prompt={_render(messages)[:200]!r})"
        )


def _render(messages: Any) -> str:
    """Cheap, stable rendering of a message list — used for substring matches."""
    if isinstance(messages, str):
        return messages
    parts: list[str] = []
    for m in messages or []:
        content = getattr(m, "content", None) or (m if isinstance(m, str) else repr(m))
        parts.append(content if isinstance(content, str) else repr(content))
    return "\n".join(parts)


def _coerce_to_message(value: Any) -> Any:
    """Wrap raw strings as a content-bearing object; pass other values through."""
    if isinstance(value, str):
        return _Plain(value)
    return value


# ---------------------------------------------------------------------------
# Convenience: schema-keyed responder
# ---------------------------------------------------------------------------


class SchemaResponder:
    """Builder for the ``by_schema`` map.

    ``SchemaResponder().for_schema(MyDecision, MyDecision(next_agent="triage", ...))``
    reads more naturally than literal dict-of-types, especially when sharing
    setup across tests.
    """

    def __init__(self) -> None:
        self._map: dict[type, Any] = {}

    def for_schema(self, schema: type, response: Any) -> "SchemaResponder":
        self._map[schema] = response
        return self

    def build(self) -> dict[type, Any]:
        return dict(self._map)


# ---------------------------------------------------------------------------
# Embeddings + retriever (placeholders for future tool-using checkpoints)
# ---------------------------------------------------------------------------


@dataclass
class FakeEmbeddings:
    """Deterministic embeddings — returns a vector derived from string length.

    Two strings with the same length produce the same vector. That's
    deliberately weak for unit tests; you don't want your test passing because
    the real embedding model happened to put two sentences close together.
    """

    dimension: int = 8

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def _vec(self, text: str) -> list[float]:
        n = len(text)
        return [(((n * (i + 1)) % 100) / 100.0) for i in range(self.dimension)]


@dataclass
class FakeRetriever:
    """Returns canned ``Document`` objects regardless of the query."""

    documents: list[Document] = field(default_factory=list)

    def get_relevant_documents(self, _query: str) -> list[Document]:
        return list(self.documents)

    async def aget_relevant_documents(self, _query: str) -> list[Document]:
        return list(self.documents)
