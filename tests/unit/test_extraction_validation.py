"""Unit tests for extraction output validation."""

import json

import pytest

from multi_rag_harness.graph.extraction import ExtractionValidationError, validate_extraction

VALID_PAYLOAD = {
    "entities": [
        {
            "name": "Qdrant",
            "type": "technology",
            "aliases": ["qdrant-client"],
            "description": "vector database",
            "confidence": 0.9,
            "evidence": "Qdrant runs in embedded mode",
        }
    ],
    "relations": [
        {
            "source": "Qdrant",
            "target": "RAG harness",
            "type": "used_by",
            "description": "",
            "confidence": 0.8,
            "evidence": "Qdrant runs in embedded mode",
            "valid_from": None,
            "valid_to": None,
        }
    ],
    "claims": [
        {
            "subject": "Qdrant",
            "predicate": "supports",
            "object": "embedded mode",
            "modality": "fact",
            "confidence": 0.95,
            "evidence": "Qdrant runs in embedded mode",
            "valid_from": None,
            "valid_to": None,
        }
    ],
}
CHUNK_TEXT = "The setup: Qdrant runs in embedded mode inside the harness."


def test_valid_payload_accepted() -> None:
    validated = validate_extraction(json.dumps(VALID_PAYLOAD), CHUNK_TEXT)
    assert validated.rejected_count == 0
    assert validated.warnings == []
    assert len(validated.result.entities) == 1
    assert len(validated.result.relations) == 1
    assert len(validated.result.claims) == 1


def test_markdown_fences_are_stripped() -> None:
    raw = "```json\n" + json.dumps(VALID_PAYLOAD) + "\n```"
    validated = validate_extraction(raw, CHUNK_TEXT)
    assert validated.rejected_count == 0


def test_invalid_json_raises() -> None:
    with pytest.raises(ExtractionValidationError):
        validate_extraction("not json at all", None)


def test_non_object_json_raises() -> None:
    with pytest.raises(ExtractionValidationError):
        validate_extraction("[1, 2, 3]", None)


def test_schema_violation_rejects_item_only() -> None:
    payload = {
        "entities": [
            VALID_PAYLOAD["entities"][0],
            {"name": "bad", "confidence": 5.0, "evidence": "Qdrant runs in embedded mode"},
        ]
    }
    validated = validate_extraction(json.dumps(payload), CHUNK_TEXT)
    assert len(validated.result.entities) == 1
    assert validated.rejected_count == 1
    assert any("schema violation" in w for w in validated.warnings)


def test_empty_evidence_rejected() -> None:
    payload = {"entities": [{**VALID_PAYLOAD["entities"][0], "evidence": "  "}]}
    validated = validate_extraction(json.dumps(payload), None)
    assert validated.result.entities == []
    assert validated.rejected_count == 1


def test_evidence_not_in_chunk_rejected() -> None:
    payload = {"entities": [{**VALID_PAYLOAD["entities"][0], "evidence": "fabricated quote"}]}
    validated = validate_extraction(json.dumps(payload), CHUNK_TEXT)
    assert validated.result.entities == []
    assert any("evidence" in w for w in validated.warnings)


def test_evidence_check_normalizes_whitespace() -> None:
    payload = {
        "entities": [{**VALID_PAYLOAD["entities"][0], "evidence": "Qdrant  runs\nin embedded mode"}]
    }
    validated = validate_extraction(json.dumps(payload), CHUNK_TEXT)
    assert len(validated.result.entities) == 1


def test_evidence_unchecked_without_chunk_text() -> None:
    payload = {"entities": [{**VALID_PAYLOAD["entities"][0], "evidence": "anything"}]}
    validated = validate_extraction(json.dumps(payload), None)
    assert len(validated.result.entities) == 1


def test_missing_keys_default_to_empty() -> None:
    validated = validate_extraction("{}", None)
    assert validated.result.entities == []
    assert validated.result.relations == []
    assert validated.result.claims == []
    assert validated.rejected_count == 0
