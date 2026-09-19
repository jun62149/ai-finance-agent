"""Qdrant vector store for financial chunks (local embedded mode by default).

Local mode (`QdrantClient(path=...)`) exposes the same API as the Qdrant
server, so moving to Docker later only changes how the client is built
(`Config.QDRANT_URL`).

Payload mirrors the chunk metadata in SQLite so retrieval results stay
traceable on their own: source / title / publish_time / url / stock_code /
doc_type / chunk_id / document_id / embedding_version.
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator, Sequence

from app.config import Config
from app.timeutil import to_market_epoch

PAYLOAD_FIELDS = (
    "chunk_id",
    "document_id",
    "stock_code",
    "stock_name",
    "doc_type",
    "title",
    "source",
    "publish_time",
    "publish_ts",
    "url",
    "chunk_index",
    "embedding_version",
)


def publish_ts_from(value: Any, *, end_of_day: bool = False) -> int | None:
    """Epoch seconds for a `publish_time` value, or None when unparsable.

    The market zone (UTC+8) is applied by `app.timeutil`, so the epoch no longer
    depends on the timezone of the process that happens to be running.
    """
    try:
        return to_market_epoch(value, end_of_day=end_of_day)
    except (ValueError, TypeError):
        return None


def _date_to_epoch(value: str, *, end_of_day: bool = False) -> int:
    """Market-zone epoch second for a YYYY-MM-DD (or ISO datetime) string.

    The payload stores `publish_ts` as an integer because Qdrant can only
    range-filter numbers. This is the *raising* entry point used when building a
    dense filter: a bad boundary must fail loudly rather than silently widen or
    narrow the window (`publish_ts_from` is the None-returning variant used when
    deriving payloads).
    """
    return to_market_epoch(value, end_of_day=end_of_day)


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
        self.ensure_payload_indexes()
        if recreate:
            # Local mode quirk (qdrant-client 1.19, measured 2026-09-10):
            # delete_collection + create_collection resurrects the old points, so
            # an explicit filter-delete is required to really empty a collection.
            self.purge()

    def ensure_payload_indexes(self) -> None:
        """Index the metadata fields used by filters (idempotent)."""
        from qdrant_client.models import PayloadSchemaType

        for field, schema in (
            ("stock_code", PayloadSchemaType.KEYWORD),
            ("doc_type", PayloadSchemaType.KEYWORD),
            ("publish_ts", PayloadSchemaType.INTEGER),
        ):
            try:
                self.client.create_payload_index(self.collection, field, schema)
            except Exception:  # noqa: BLE001 - already exists / unsupported locally
                continue

    def set_payloads(self, point_ids: Sequence[int], payloads: Sequence[dict[str, Any]]) -> int:
        """Update payload fields in place (no re-embedding)."""
        if not (len(point_ids) == len(payloads)):
            raise ValueError("point_ids and payloads must have equal length")
        updated = 0
        for point_id, payload in zip(point_ids, payloads):
            self.client.set_payload(self.collection, payload=payload, points=[int(point_id)])
            updated += 1
        return updated

    def purge(self) -> int:
        """Delete every point in the collection, keeping the collection itself."""
        from qdrant_client.models import Filter, FilterSelector

        before = self.count()
        self.client.delete(
            self.collection, points_selector=FilterSelector(filter=Filter())
        )
        return before - self.count()

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
            points = []
            for i in range(start, min(end, len(point_ids))):
                payload = {key: payloads[i].get(key) for key in PAYLOAD_FIELDS}
                # Derive publish_ts here too: it is what date filters range over,
                # and a point without it is invisible to any date-filtered query.
                if payload.get("publish_ts") is None:
                    payload["publish_ts"] = publish_ts_from(payload.get("publish_time"))
                points.append(
                    PointStruct(
                        id=int(point_ids[i]),
                        vector=list(map(float, vectors[i])),
                        payload=payload,
                    )
                )
            self.client.upsert(self.collection, points=points, wait=True)
            total += len(points)
        return total

    # -- reads -------------------------------------------------------------- #
    def scroll_points(
        self, batch_size: int = 256
    ) -> Iterator[tuple[list[int], list[list[float]], list[dict[str, Any]]]]:
        """Iterate over every stored point (ids, vectors, payloads).

        Used to move a collection between Qdrant targets (local path mode ->
        Docker server) without re-running the embedding model.
        """
        offset: Any = None
        while True:
            points, offset = self.client.scroll(
                self.collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            if not points:
                return
            ids: list[int] = []
            vectors: list[list[float]] = []
            payloads: list[dict[str, Any]] = []
            for point in points:
                ids.append(int(point.id))
                vector = point.vector
                if isinstance(vector, dict):  # named vectors: take the first
                    vector = next(iter(vector.values()))
                vectors.append(list(map(float, vector)))
                payloads.append(dict(point.payload or {}))
            yield ids, vectors, payloads
            if offset is None:
                return

    def search(
        self,
        vector: Sequence[float],
        top_k: int = 5,
        stock_code: str | None = None,
        doc_type: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[dict[str, Any]]:
        query_filter = self._build_filter(
            stock_code=stock_code, doc_type=doc_type,
            date_from=date_from, date_to=date_to,
        )
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
    def _build_filter(
        stock_code: str | None,
        doc_type: str | None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> Any:
        if not (stock_code or doc_type or date_from or date_to):
            return None
        from qdrant_client.models import FieldCondition, Filter, MatchValue, Range

        conditions = []
        if stock_code:
            conditions.append(
                FieldCondition(key="stock_code", match=MatchValue(value=stock_code))
            )
        if doc_type:
            conditions.append(
                FieldCondition(key="doc_type", match=MatchValue(value=doc_type))
            )
        if date_from or date_to:
            conditions.append(
                FieldCondition(
                    key="publish_ts",
                    range=Range(
                        gte=_date_to_epoch(date_from) if date_from else None,
                        lte=_date_to_epoch(date_to, end_of_day=True) if date_to else None,
                    ),
                )
            )
        return Filter(must=conditions)
