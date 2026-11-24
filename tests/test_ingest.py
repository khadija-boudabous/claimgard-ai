import uuid
from pathlib import Path

import pytest

from app.rag.ingest import Chunk, load_chunks, parse_markdown, split_paragraphs

EXTRACT = """# Conditions Générales — Assurance Auto

> Document fictif à usage de démonstration.

## Titre I — Garanties

### Article 1 — Bris de glace

Le pare-brise est couvert sans franchise.

Les vitres latérales sont également couvertes.

### Article 2 — Vol

Le vol avec effraction est couvert.
"""


@pytest.fixture
def contract_file(tmp_path: Path) -> Path:
    path = tmp_path / "cg_auto.md"
    path.write_text(EXTRACT, encoding="utf-8")
    return path


def test_split_paragraphs_keeps_short_body_in_one_chunk() -> None:
    assert split_paragraphs("Un.\n\nDeux.\n\n\n", max_words=50) == ["Un.\n\nDeux."]


def test_split_paragraphs_overlaps_one_paragraph() -> None:
    body = "\n\n".join(["a b c", "d e f", "g h i"])

    chunks = split_paragraphs(body, max_words=6)

    assert chunks == ["a b c\n\nd e f", "d e f\n\ng h i"]


def test_split_paragraphs_ignores_blank_body() -> None:
    assert split_paragraphs("  \n\n  ", max_words=10) == []


def test_parse_markdown_extracts_hierarchy(contract_file: Path) -> None:
    articles = list(parse_markdown(contract_file))

    assert [(c, s, a) for c, s, a, _ in articles] == [
        ("Assurance Auto", "Titre I — Garanties", "Article 1 — Bris de glace"),
        ("Assurance Auto", "Titre I — Garanties", "Article 2 — Vol"),
    ]
    assert all("Document fictif" not in body for *_, body in articles)


def test_chunk_ids_are_deterministic_across_runs(contract_file: Path) -> None:
    first = load_chunks(contract_file, max_words=150)
    second = load_chunks(contract_file, max_words=150)

    assert [c.id for c in first] == [c.id for c in second]
    assert len({c.id for c in first}) == len(first)
    assert first[0].contract_type == "auto"
    assert first[0].embedding_text().startswith("[Assurance Auto — Titre I — Garanties — Article 1 — Bris de glace]\n")


def test_chunk_id_depends_on_source_article_and_index() -> None:
    chunk = Chunk("texte", "Auto", "auto", "Titre I", "Article 1", 0, "cg_auto.md")

    assert chunk.id == str(uuid.uuid5(uuid.NAMESPACE_URL, "cg_auto.md|Article 1|0"))
    assert chunk.id != Chunk("texte", "Auto", "auto", "Titre I", "Article 1", 1, "cg_auto.md").id
    assert chunk.id != Chunk("texte", "Auto", "auto", "Titre I", "Article 1", 0, "cg_habitation.md").id
    # Le texte n'entre pas dans l'ID : une clause corrigée écrase son ancienne version (upsert)
    assert chunk.id == Chunk("texte v2", "Auto", "auto", "Titre I", "Article 1", 0, "cg_auto.md").id
