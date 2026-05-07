"""Testing helpers — public surface that test files import from.

Three modules:

- ``fakes`` — deterministic stand-ins for ChatModel/Embeddings/Retriever.
- ``recorders`` — capture trajectory and state snapshots from running graphs.
- ``fixtures`` — pytest fixtures that wire the fakes/recorders into tests.

The split is intentional: ``fakes`` and ``recorders`` are pytest-free so
non-pytest tooling (the replay CLI) can use them too.
"""

from app.testing.fakes import (
    FakeChatModel,
    FakeEmbeddings,
    FakeRetriever,
    SchemaResponder,
)
from app.testing.recorders import StateRecorder, TrajectoryRecorder, TrajectoryStep

__all__ = [
    "FakeChatModel",
    "FakeEmbeddings",
    "FakeRetriever",
    "SchemaResponder",
    "StateRecorder",
    "TrajectoryRecorder",
    "TrajectoryStep",
]
