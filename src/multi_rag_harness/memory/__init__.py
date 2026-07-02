"""Durable memory package for traces, decisions, failures, code, and tools.

Each memory kind persists a typed table row (the authoritative record) and a
rendered markdown document indexed through the ingestion pipeline, so all
memory is retrievable via the one hybrid search contract.
"""

from __future__ import annotations

from pydantic import BaseModel


class StoredMemoryRef(BaseModel):
    """Reference returned by memory stores: typed record + search document."""

    record_id: str
    document_id: str


def render_section(title: str, body: str | None) -> str:
    if body is None or not str(body).strip():
        return ""
    return f"## {title}\n\n{body}\n\n"


def render_list_section(title: str, items: list[str]) -> str:
    if not items:
        return ""
    bullets = "\n".join(f"- {item}" for item in items)
    return f"## {title}\n\n{bullets}\n\n"
