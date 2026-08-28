"""Batched embeddings via ``POST /embeddings``; order-preserving, dimension-checked."""

from __future__ import annotations

from .llm_http import LLMClient


class EmbeddingError(RuntimeError):
    pass


def embed_texts(client: LLMClient, model: str, texts: list[str], *, batch_size: int, dim: int,
                prefix: str = "") -> list[list[float]]:
    out: list[list[float] | None] = [None] * len(texts)
    for start in range(0, len(texts), max(1, batch_size)):
        batch = texts[start:start + batch_size]
        resp = client.post_json("/embeddings", {"model": model, "input": [prefix + t for t in batch]})
        data = resp.get("data") or []
        if len(data) != len(batch):
            raise EmbeddingError(f"expected {len(batch)} embeddings, got {len(data)}")
        for item in data:
            idx = int(item["index"])
            vec = item["embedding"]
            if not 0 <= idx < len(batch):
                raise EmbeddingError(f"embedding index {idx} out of range")
            if len(vec) != dim:
                raise EmbeddingError(f"embedding dim {len(vec)} != configured {dim}")
            out[start + idx] = [float(x) for x in vec]
    missing = [i for i, v in enumerate(out) if v is None]
    if missing:
        raise EmbeddingError(f"{len(missing)} embeddings missing (first: {missing[:3]})")
    return out  # type: ignore[return-value]
