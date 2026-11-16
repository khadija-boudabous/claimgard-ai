"""Embeddings hybrides CPU (dense multilingue + BM25), partagés par l'ingestion et la recherche."""

from dataclasses import dataclass

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import models

from app.config import Settings

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"


@dataclass(frozen=True)
class HybridEmbedder:
    dense: TextEmbedding
    sparse: SparseTextEmbedding
    dense_size: int

    def embed_documents(
        self, texts: list[str]
    ) -> tuple[list[list[float]], list[models.SparseVector]]:
        dense = [vec.tolist() for vec in self.dense.embed(texts)]
        sparse = [_to_sparse(emb) for emb in self.sparse.embed(texts)]
        return dense, sparse

    def embed_query(self, text: str) -> tuple[list[float], models.SparseVector]:
        dense = next(iter(self.dense.query_embed(text))).tolist()
        sparse = _to_sparse(next(iter(self.sparse.query_embed(text))))
        return dense, sparse


def _to_sparse(embedding: object) -> models.SparseVector:
    return models.SparseVector(
        indices=embedding.indices.tolist(),  # type: ignore[attr-defined]
        values=embedding.values.tolist(),  # type: ignore[attr-defined]
    )


def load_embedder(settings: Settings) -> HybridEmbedder:
    """Charge (et télécharge au 1er appel) les deux modèles ONNX dans le cache persistant."""
    dense = TextEmbedding(settings.dense_model, cache_dir=settings.embed_cache_dir)
    sparse = SparseTextEmbedding(
        settings.sparse_model,
        cache_dir=settings.embed_cache_dir,
        language=settings.sparse_language,
    )
    dense_size = TextEmbedding.get_embedding_size(settings.dense_model)
    return HybridEmbedder(dense=dense, sparse=sparse, dense_size=dense_size)
