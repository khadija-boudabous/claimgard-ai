"""Ingestion des conditions générales : parsing → chunking → embeddings hybrides → Qdrant.

Usage : python -m app.rag.ingest [--recreate] [--data-dir data/contracts]
"""

import argparse
import logging
import re
import uuid
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from pypdf import PdfReader
from qdrant_client import QdrantClient, models

from app.config import Settings, get_settings
from app.rag.embedder import DENSE_VECTOR, SPARSE_VECTOR, HybridEmbedder, load_embedder

logger = logging.getLogger("ingest")

BATCH_SIZE = 64
ARTICLE_RE = re.compile(r"(?m)^\s*(Article\s+\d+[^\n]*)$")


@dataclass(frozen=True)
class Chunk:
    text: str
    contract: str
    contract_type: str
    section: str
    article: str
    chunk_index: int
    source: str

    @property
    def id(self) -> str:
        """ID déterministe → ré-ingestion idempotente (upsert sans doublon)."""
        key = f"{self.source}|{self.article}|{self.chunk_index}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, key))

    def embedding_text(self) -> str:
        """Préfixe contextuel : situe la clause dans son contrat pour l'embedding."""
        header = " — ".join(p for p in (self.contract, self.section, self.article) if p)
        return f"[{header}]\n{self.text}"


def split_paragraphs(body: str, max_words: int) -> list[str]:
    """Regroupe les paragraphes jusqu'à max_words, avec 1 paragraphe de chevauchement."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    for para in paragraphs:
        candidate = current + [para]
        if current and sum(len(p.split()) for p in candidate) > max_words:
            chunks.append("\n\n".join(current))
            current = [current[-1], para]
        else:
            current = candidate
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _heading_title(line: str) -> str:
    """'# Conditions Générales — Assurance Auto' → 'Assurance Auto'."""
    title = line.lstrip("#").strip()
    return title.split(" — ", 1)[1] if title.startswith("Conditions Générales — ") else title


def parse_markdown(path: Path) -> Iterator[tuple[str, str, str, str]]:
    """Produit (contrat, titre, article, corps) pour chaque article `### …`."""
    contract, section, article = path.stem, "", ""
    body: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            if article and body:
                yield contract, section, article, "\n".join(body)
            body = []
            level = len(line) - len(line.lstrip("#"))
            if level == 1:
                contract = _heading_title(line)
            elif level == 2:
                section, article = line.lstrip("#").strip(), ""
            else:
                article = line.lstrip("#").strip()
        elif not line.startswith(">"):  # ignore les bandeaux/avertissements
            body.append(line)
    if article and body:
        yield contract, section, article, "\n".join(body)


def parse_pdf(path: Path) -> Iterator[tuple[str, str, str, str]]:
    """PDF : texte brut découpé sur les intitulés « Article N » (sans hiérarchie de titres)."""
    text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    parts = ARTICLE_RE.split(text)
    for article, body in zip(parts[1::2], parts[2::2], strict=True):
        yield path.stem, "", article.strip(), body


def load_chunks(path: Path, max_words: int) -> list[Chunk]:
    parser = parse_pdf if path.suffix.lower() == ".pdf" else parse_markdown
    contract_type = path.stem.removeprefix("cg_")
    return [
        Chunk(text, contract, contract_type, section, article, i, path.name)
        for contract, section, article, body in parser(path)
        for i, text in enumerate(split_paragraphs(body, max_words))
    ]


def ensure_collection(client: QdrantClient, name: str, dense_size: int, recreate: bool) -> None:
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
        logger.info("Collection '%s' supprimée", name)
    if client.collection_exists(name):
        return
    client.create_collection(
        name,
        vectors_config={
            DENSE_VECTOR: models.VectorParams(size=dense_size, distance=models.Distance.COSINE)
        },
        # fastembed/bm25 ne fournit que les TF : Qdrant calcule l'IDF sur le corpus
        sparse_vectors_config={SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    logger.info("Collection '%s' créée (dense=%d dim + bm25)", name, dense_size)


def upsert_chunks(
    client: QdrantClient, collection: str, embedder: HybridEmbedder, chunks: list[Chunk]
) -> None:
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        dense, sparse = embedder.embed_documents([c.embedding_text() for c in batch])
        points = [
            models.PointStruct(
                id=chunk.id,
                vector={DENSE_VECTOR: d, SPARSE_VECTOR: s},
                payload=asdict(chunk),
            )
            for chunk, d, s in zip(batch, dense, sparse, strict=True)
        ]
        client.upsert(collection, points=points, wait=True)


def run(settings: Settings, data_dir: Path, recreate: bool) -> int:
    files = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in {".md", ".pdf"})
    chunks = [c for path in files for c in load_chunks(path, settings.chunk_max_words)]
    for source, count in Counter(c.source for c in chunks).items():
        logger.info("%s : %d chunks", source, count)

    embedder = load_embedder(settings)
    client = QdrantClient(url=str(settings.qdrant_url))
    try:
        ensure_collection(client, settings.qdrant_collection, embedder.dense_size, recreate)
        upsert_chunks(client, settings.qdrant_collection, embedder, chunks)
        total = client.count(settings.qdrant_collection, exact=True).count
    finally:
        client.close()
    logger.info("%d chunks indexés — collection '%s' : %d points", len(chunks), settings.qdrant_collection, total)
    return total


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Indexe les CG d'assurance dans Qdrant.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/contracts"))
    parser.add_argument("--recreate", action="store_true", help="supprime et recrée la collection")
    args = parser.parse_args()
    run(get_settings(), args.data_dir, args.recreate)


if __name__ == "__main__":
    main()
