"""Second étage du retrieval : cross-encoder qui re-score les candidats issus de la fusion RRF."""

from fastembed.rerank.cross_encoder import TextCrossEncoder

from app.config import Settings


def load_reranker(settings: Settings) -> TextCrossEncoder | None:
    """Charge (et télécharge au 1er appel) le cross-encoder ONNX ; None = reranking désactivé."""
    if not settings.rerank_model:
        return None
    return TextCrossEncoder(settings.rerank_model, cache_dir=settings.embed_cache_dir)


def rerank(encoder: TextCrossEncoder, query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
    """Retourne les (index, score) des top_k documents, du plus au moins pertinent."""
    scores = [float(s) for s in encoder.rerank(query, documents)]
    ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
    return ranked[:top_k]
