"""Unit tests for extraction prompt and pinned schema compatibility."""

from typing import get_args

from multi_rag_harness.codex.prompts import (
    EXTRACTION_SCHEMA,
    PROMPT_VERSION,
    build_extraction_prompt,
)
from multi_rag_harness.graph.models import (
    ExtractedClaim,
    ExtractedEntity,
    ExtractedRelation,
    Modality,
)


def test_prompt_version() -> None:
    assert PROMPT_VERSION == "extraction/v1"


def test_prompt_contains_chunk_and_context() -> None:
    prompt = build_extraction_prompt(
        "Kuzu stores the graph.", heading_path="Notes > Graph", document_title="Arch"
    )
    assert "Kuzu stores the graph." in prompt
    assert "Document: Arch" in prompt
    assert "Section: Notes > Graph" in prompt
    assert "JSON only" in prompt


def test_schema_top_level_shape() -> None:
    assert EXTRACTION_SCHEMA["required"] == ["entities", "relations", "claims"]
    assert set(EXTRACTION_SCHEMA["properties"]) == {"entities", "relations", "claims"}


def test_schema_modality_enum_matches_model() -> None:
    enum = EXTRACTION_SCHEMA["properties"]["claims"]["items"]["properties"]["modality"]["enum"]
    assert set(enum) == set(get_args(Modality))


def test_schema_required_fields_are_accepted_by_models() -> None:
    """Everything the pinned schema marks required must be sufficient for the
    pydantic models (i.e. models must not require more than the schema)."""
    entity_required = EXTRACTION_SCHEMA["properties"]["entities"]["items"]["required"]
    ExtractedEntity.model_validate(
        {key: 0.5 if key == "confidence" else "x" for key in entity_required}
    )
    relation_required = EXTRACTION_SCHEMA["properties"]["relations"]["items"]["required"]
    ExtractedRelation.model_validate(
        {key: 0.5 if key == "confidence" else "x" for key in relation_required}
    )
    claim_required = EXTRACTION_SCHEMA["properties"]["claims"]["items"]["required"]
    ExtractedClaim.model_validate(
        {key: 0.5 if key == "confidence" else "x" for key in claim_required}
    )


def test_schema_properties_cover_model_fields() -> None:
    for key, model in (
        ("entities", ExtractedEntity),
        ("relations", ExtractedRelation),
        ("claims", ExtractedClaim),
    ):
        schema_props = set(EXTRACTION_SCHEMA["properties"][key]["items"]["properties"])
        model_fields = set(model.model_fields)
        assert schema_props == model_fields, f"{key} schema/model field mismatch"
