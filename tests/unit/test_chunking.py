"""Unit tests for chunking strategies."""

import pytest

from multi_rag_harness.ingestion.chunking import (
    chunk_code,
    chunk_markdown,
    chunk_text,
    estimate_tokens,
)


def test_estimate_tokens_ascii() -> None:
    assert estimate_tokens("abcd" * 10) == 10


def test_estimate_tokens_cjk_counts_per_codepoint() -> None:
    assert estimate_tokens("日本語テスト") == 6


def test_estimate_tokens_mixed() -> None:
    # 2 CJK chars + 4 ascii chars -> 2 + ceil(4/4) = 3
    assert estimate_tokens("日本abcd") == 3


def test_chunk_markdown_heading_paths_stack() -> None:
    text = (
        "# Title\n\nintro paragraph\n\n"
        "## Setup\n\nsetup body\n\n"
        "### Install\n\ninstall body\n\n"
        "## Usage\n\nusage body\n"
    )
    drafts = chunk_markdown(text)
    paths = [d.heading_path for d in drafts]
    assert paths == ["Title", "Title > Setup", "Title > Setup > Install", "Title > Usage"]
    assert [d.ordinal for d in drafts] == [0, 1, 2, 3]
    assert drafts[1].text == "setup body"


def test_chunk_markdown_packs_paragraphs_up_to_budget() -> None:
    paragraphs = "\n\n".join(f"paragraph number {i} " + "word " * 20 for i in range(10))
    drafts = chunk_markdown("# H\n\n" + paragraphs, max_tokens=60)
    assert len(drafts) > 1
    assert all(d.token_count <= 60 * 2 for d in drafts)
    # All content preserved.
    combined = "\n\n".join(d.text for d in drafts)
    for i in range(10):
        assert f"paragraph number {i}" in combined


def test_chunk_markdown_splits_oversized_paragraph_at_lines() -> None:
    big_paragraph = "\n".join("line " + "x " * 30 for _ in range(20))
    drafts = chunk_markdown(big_paragraph, max_tokens=50)
    assert len(drafts) > 1


def test_chunk_text_has_no_heading_paths() -> None:
    drafts = chunk_text("first para\n\nsecond para")
    assert len(drafts) == 1
    assert drafts[0].heading_path == ""


def test_chunk_code_windows_and_overlap() -> None:
    lines = "\n".join(f"line{i}" for i in range(300))
    drafts = chunk_code(lines, max_lines=120, overlap=20)
    assert len(drafts) == 3
    assert drafts[0].text.startswith("line0")
    # Overlap: window 2 starts 100 lines in.
    assert drafts[1].text.startswith("line100")
    assert drafts[2].text.endswith("line299")


def test_chunk_code_empty() -> None:
    assert chunk_code("") == []


def test_chunk_code_overlap_validation() -> None:
    with pytest.raises(ValueError):
        chunk_code("x", max_lines=10, overlap=10)
