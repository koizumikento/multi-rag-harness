"""pgvector storage adapter (not yet implemented).

Selectable via ``storage.vector_backend = "pgvector"``; every operation raises
``NotImplementedError``. The pgvector dependency is kept for this future
adapter.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_MESSAGE = "pgvector backend not implemented; use vector_backend='qdrant'"


class PgvectorIndex:
    """Placeholder VectorIndex over Postgres + pgvector."""

    def __init__(self, dsn: str | None) -> None:
        self._dsn = dsn

    async def initialize(self, dimension: int) -> None:
        raise NotImplementedError(_MESSAGE)

    async def close(self) -> None:
        return None

    def __getattr__(self, name: str) -> Callable[..., Any]:
        async def _unimplemented(*args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError(_MESSAGE)

        return _unimplemented
