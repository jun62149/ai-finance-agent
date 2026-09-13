"""Qdrant vector store for financial chunks (local embedded mode by default).

Local mode (`QdrantClient(path=...)`) exposes the same API as the Qdrant
server, so moving to Docker later only changes how the client is built
(`Config.QDRANT_URL`).

Payload mirrors the chunk metadata in SQLite so retrieval results stay
traceable on their own: source / title / publish_time / url / stock_code /
doc_type / chunk_id / document_id / embedding_version.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from app.config import Config

PAYLOAD_FIELDS = (
    "chunk_id",
    "document_id",
    "stock_code",
    "stock_name",
    "doc_type",
    "title",
    "source",
    "publish_time",
    "url",
    "chunk_index",
    "embedding_version",
)


class ChunkVectorStore:
    """Thin Qdrant wrapper used by the indexing and search paths."""

    def __init__(
        self,
        path: str | None = None,
        url: str | None = None,
        collection: str | None = None,
        dim: int | None = None,
        client: Any = None,
    ) -> None:
        self.collection = collection or Config.QDRANT_COLLECTION
        self.dim = dim or Config.EMBEDDING_DIM
        self._owns_client = client is None
        if client is not None:
            self.client = client
        else:
            self.client = self._build_client(path, url)

    @staticmethod
    def _build_client(path: str | None, url: str | None) -> Any:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError(
                "qdrant-client is required for the vector store; install it with: "
                "pip install qdrant-client"
            ) from exc
        target_url = url if url is not None else Config.QDRANT_URL
        if target_url:
            return QdrantClient(url=target_url)
        target_path = str(path or Config.QDRANT_PATH)
        return QdrantClient(path=target_path)

    # -- collection --------------------------------------------------------- #
    def ensure_collection(self, recreate: bool = False) -> None:
        from qdrant_client.models import Distance, VectorParams

        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if not exists:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )

    def count(self) -> int:
        return int(self.client.count(self.collection, exact=True).count)

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> "ChunkVectorStore":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # -- writes ------------------------------------------------------------- #
    def upsert_chunks(
        self,
        point_ids: Sequence[int],
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[dict[str, Any]],
        batch_size: int = 256,
    ) -> int:
        from qdrant_client.models import PointStruct

        if not (len(point_ids) == len(vectors) == len(payloads)):
            raise ValueError("point_ids, vectors and payloads must have equal length")
        total = 0
        for start in range(0, len(point_ids), batch_size):
            end = start + batch_size
            points = [
                PointStruct(
                    id=int(point_ids[i]),
                    vector=list(map(float, vectors[i])),
                    payload={
                        key: payloads[i].get(key) for key in PAYLOAD_FIELDS
                    },
                )
                for i in range(start, min(end, len(point_ids)))
            ]
            self.client.upsert(self.collection, points=points, wait=True)
            total += len(points)
        return total

    # -- reads -------------------------------------------------------------- #
    def search(
        self,
        vector: Sequence[float],
        top_k: int = 5,
        stock_code: str | None = None,
        doc_type: str | None = None,
    ) -> list[dict[str, Any]]:
        query_filter = self._build_filter(stock_code=stock_code, doc_type=doc_type)
        response = self.client.query_points(
            self.collection,
            query=list(map(float, vector)),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        results = []
        for point in response.points:
            payload = dict(point.payload or {})
            payload["score"] = float(point.score)
            payload["point_id"] = int(point.id)
            results.append(payload)
        return results

    @staticmethod
    def _build_filter(stock_code: str | None, doc_type: str | None) -> Any:
        if not stock_code and not doc_type:
            return None
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        conditions = []
        if stock_code:
            conditions.append(
                FieldCondition(key="stock_code", match=MatchValue(value=stock_code))
            )
        if doc_type:
            conditions.append(
                FieldCondition(key="doc_type", match=MatchValue(value=doc_type))
            )
        return Filter(must=conditions)
