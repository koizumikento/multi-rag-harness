"""Unit tests for memory payload rendering."""

from multi_rag_harness.memory.decisions import DecisionPayload
from multi_rag_harness.memory.failures import FailurePayload
from multi_rag_harness.memory.tools import ToolRecordPayload
from multi_rag_harness.memory.traces import TracePayload


def test_trace_render_contains_key_fields() -> None:
    payload = TracePayload(
        task="fix flaky test",
        outcome="success",
        tools_used=["pytest"],
        commands=["uv run pytest"],
        errors=["AssertionError"],
    )
    text = payload.render_text()
    assert "# Trace: fix flaky test" in text
    assert "## Outcome" in text and "success" in text
    assert "## Tools Used" in text and "- pytest" in text
    assert "## Errors" in text and "- AssertionError" in text
    # Empty optional sections are omitted.
    assert "## Human Feedback" not in text


def test_decision_render_contains_key_fields() -> None:
    payload = DecisionPayload(
        title="Use SQLite",
        decision="SQLite is the metadata backend",
        context="local-first requirement",
        rationale="no server needed",
        alternatives=["Postgres"],
    )
    text = payload.render_text()
    assert "# Decision: Use SQLite" in text
    assert "## Decision" in text and "SQLite is the metadata backend" in text
    assert "## Rationale" in text
    assert "- Postgres" in text


def test_failure_render_contains_key_fields() -> None:
    payload = FailurePayload(
        error_text="ImportError: no module named foo",
        error_category="import",
        fix_applied="added dependency",
        related_code_paths=["src/foo.py"],
    )
    text = payload.render_text()
    assert "ImportError: no module named foo" in text
    assert "## Fix Applied" in text
    assert "- src/foo.py" in text


def test_tool_render_contains_key_fields() -> None:
    payload = ToolRecordPayload(
        server="multi-rag-harness",
        name="rag_search",
        description="hybrid search over docs",
        input_schema={"type": "object"},
        known_failure_modes=["empty query raises"],
    )
    text = payload.render_text()
    assert "# Tool: multi-rag-harness/rag_search" in text
    assert "hybrid search over docs" in text
    assert '"type": "object"' in text
    assert "- empty query raises" in text
