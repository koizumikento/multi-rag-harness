"""Prompt templates for Codex SDK orchestration.

``EXTRACTION_SCHEMA`` is hand-pinned to the JSON schema in the specification
(not generated from the Pydantic models) so prompt-visible structure stays
stable across pydantic upgrades; a unit test guards compatibility with
``graph.models.ExtractionResult``.
"""

from __future__ import annotations

from typing import Any

PROMPT_VERSION = "extraction/v1"

_CONFIDENCE = {"type": "number", "minimum": 0.0, "maximum": 1.0}
_NULLABLE_STRING = {"type": ["string", "null"]}

EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                    "description": {"type": "string"},
                    "confidence": _CONFIDENCE,
                    "evidence": {"type": "string"},
                },
                "required": ["name", "confidence", "evidence"],
                "additionalProperties": False,
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "type": {"type": "string"},
                    "description": {"type": "string"},
                    "confidence": _CONFIDENCE,
                    "evidence": {"type": "string"},
                    "valid_from": _NULLABLE_STRING,
                    "valid_to": _NULLABLE_STRING,
                },
                "required": ["source", "target", "type", "confidence", "evidence"],
                "additionalProperties": False,
            },
        },
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    "modality": {
                        "type": "string",
                        "enum": ["fact", "hypothesis", "decision", "requirement", "constraint"],
                    },
                    "confidence": _CONFIDENCE,
                    "evidence": {"type": "string"},
                    "valid_from": _NULLABLE_STRING,
                    "valid_to": _NULLABLE_STRING,
                },
                "required": ["subject", "predicate", "object", "confidence", "evidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["entities", "relations", "claims"],
    "additionalProperties": False,
}

_EXTRACTION_INSTRUCTIONS = """\
You are a precise knowledge-graph extraction engine. Extract entities,
relations, and claims from the source text below.

Rules:
- Extract only what the text states or directly implies; do not invent facts.
- `evidence` must be a verbatim span copied from the source text.
- Use short canonical names for entities; put surface variants in `aliases`.
- Relation `source` and `target` must be entity names from your `entities`
  list whenever possible.
- Claims capture statements about a subject: pick the best `modality`
  (fact | hypothesis | decision | requirement | constraint).
- Set `confidence` between 0 and 1.
- Use ISO-8601 dates for `valid_from` / `valid_to` when the text gives
  temporal bounds; otherwise null.
- Respond with JSON only, matching the required schema. No commentary.
"""


def build_extraction_prompt(
    chunk_text: str, heading_path: str = "", document_title: str = ""
) -> str:
    context_lines = []
    if document_title:
        context_lines.append(f"Document: {document_title}")
    if heading_path:
        context_lines.append(f"Section: {heading_path}")
    context = ("\n".join(context_lines) + "\n\n") if context_lines else ""
    return f"{_EXTRACTION_INSTRUCTIONS}\n{context}Source text:\n---\n{chunk_text}\n---"
