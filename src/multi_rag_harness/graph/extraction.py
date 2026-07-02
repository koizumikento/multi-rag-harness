"""Graph extraction result validation.

Pipeline: JSON parse (markdown fences stripped) → per-item schema validation →
evidence presence check. Invalid items are rejected individually with
warnings; the rest are kept.
"""

from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import ValidationError

from multi_rag_harness.graph.models import (
    ExtractedClaim,
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
    ValidatedExtraction,
)

_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n(.*)\n?```\s*$", re.DOTALL)

ItemT = TypeVar("ItemT", ExtractedEntity, ExtractedRelation, ExtractedClaim)


class ExtractionValidationError(ValueError):
    """Raised when extraction output cannot be parsed at all."""


def _strip_fences(raw: str) -> str:
    stripped = raw.strip()
    match = _FENCE_RE.match(stripped)
    return match.group(1) if match else stripped


def _normalize_ws(text: str) -> str:
    return " ".join(text.split())


def _evidence_ok(evidence: str, chunk_text: str | None) -> bool:
    if not evidence.strip():
        return False
    if chunk_text is None:
        return True
    return _normalize_ws(evidence) in _normalize_ws(chunk_text)


def validate_extraction(raw: str, chunk_text: str | None = None) -> ValidatedExtraction:
    """Validate raw Codex extraction output against the fixed schema.

    Raises :class:`ExtractionValidationError` when the payload is not a JSON
    object. Individual invalid items (schema violations, missing or
    non-grounded evidence) are dropped with warnings.
    """
    try:
        payload = json.loads(_strip_fences(raw))
    except json.JSONDecodeError as exc:
        raise ExtractionValidationError(f"extraction output is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ExtractionValidationError("extraction output must be a JSON object")

    warnings: list[str] = []
    rejected = 0

    def validate_items(key: str, model: type[ItemT]) -> list[ItemT]:
        nonlocal rejected
        items = payload.get(key, [])
        if not isinstance(items, list):
            warnings.append(f"{key}: expected a list, got {type(items).__name__}")
            rejected += 1
            return []
        accepted: list[ItemT] = []
        for index, item in enumerate(items):
            try:
                parsed = model.model_validate(item)
            except ValidationError as exc:
                warnings.append(f"{key}[{index}]: schema violation: {exc.error_count()} errors")
                rejected += 1
                continue
            if not _evidence_ok(parsed.evidence, chunk_text):
                warnings.append(f"{key}[{index}]: evidence missing or not found in chunk")
                rejected += 1
                continue
            accepted.append(parsed)
        return accepted

    result = ExtractionResult(
        entities=validate_items("entities", ExtractedEntity),
        relations=validate_items("relations", ExtractedRelation),
        claims=validate_items("claims", ExtractedClaim),
    )
    return ValidatedExtraction(result=result, warnings=warnings, rejected_count=rejected)
