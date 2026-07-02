"""Postgres storage adapter (not yet implemented).

Selectable via ``storage.metadata_backend = "postgres"``; every operation
raises ``NotImplementedError``. The sqlalchemy/psycopg dependencies are kept
for this future adapter.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_MESSAGE = "postgres backend not implemented; use metadata_backend='sqlite'"


class PostgresStore:
    """Placeholder MetadataStore + KeywordIndex over Postgres."""

    def __init__(self, dsn: str | None) -> None:
        self._dsn = dsn

    async def initialize(self) -> None:
        raise NotImplementedError(_MESSAGE)

    async def close(self) -> None:
        return None

    def __getattr__(self, name: str) -> Callable[..., Any]:
        async def _unimplemented(*args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError(_MESSAGE)

        return _unimplemented
