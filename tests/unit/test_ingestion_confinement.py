"""HTTP corpus authorization must apply before any linked file is loaded."""

from pathlib import Path

import pytest

from multi_rag_harness.ingestion.documents import iter_documents


def test_linked_files_cannot_escape_http_corpus(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    secret = tmp_path / "secret.md"
    secret.write_text("not authorized", encoding="utf-8")
    link = corpus / "link.md"
    try:
        link.symlink_to(secret)
    except OSError:
        pytest.skip("This Windows account cannot create symlinks; Linux CI covers it")
    with pytest.raises(ValueError, match="outside"):
        list(iter_documents(corpus, allowed_root=corpus.resolve()))
    with pytest.raises(ValueError, match="inside"):
        list(iter_documents(link, allowed_root=corpus.resolve()))
    assert list(iter_documents(link))[0].text == "not authorized"  # Local behavior preserved.


def test_inside_corpus_file_and_missing_file(tmp_path: Path):
    file = tmp_path / "note.md"
    file.write_text("# Note\nfixture", encoding="utf-8")
    assert list(iter_documents(file, allowed_root=tmp_path.resolve()))[0].title == "Note"
    with pytest.raises(FileNotFoundError):
        list(iter_documents(tmp_path / "missing.md", allowed_root=tmp_path.resolve()))
