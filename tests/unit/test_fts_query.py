"""Unit tests for FTS5 MATCH expression building."""

import pytest

from multi_rag_harness.storage.sqlite import build_fts_match


def test_tokens_are_quoted_and_or_joined() -> None:
    assert build_fts_match("hello world") == '"hello" OR "world"'


def test_operators_are_neutralized() -> None:
    assert build_fts_match("NOT AND OR") == '"NOT" OR "AND" OR "OR"'
    assert build_fts_match("NEAR(foo") == '"NEAR(foo"'
    assert build_fts_match("wild*") == '"wild*"'
    assert build_fts_match("-negated") == '"-negated"'
    assert build_fts_match("col:value") == '"col:value"'
    assert build_fts_match("a^b") == '"a^b"'


def test_double_quotes_are_escaped() -> None:
    assert build_fts_match('say "hi"') == '"say" OR """hi"""'


def test_cjk_query_without_spaces_is_single_phrase() -> None:
    assert build_fts_match("検索エラー") == '"検索エラー"'


def test_error_message_with_path() -> None:
    match = build_fts_match("ImportError: /usr/lib/foo.py")
    assert match == '"ImportError:" OR "/usr/lib/foo.py"'


def test_empty_query_raises() -> None:
    with pytest.raises(ValueError):
        build_fts_match("")
    with pytest.raises(ValueError):
        build_fts_match("   ")
