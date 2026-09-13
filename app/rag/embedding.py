"""Embedding layer (BGE-M3) for financial chunks.

Design notes:
- The model name, device, batch size and version come from Config / env, never
  hard-coded, so swapping the embedding model is a configuration change.
- `EMBEDDING_VERSION` is stored on every chunk it produced. Vectors from two
  different versions must never be mixed in one collection (otherwise recall
  behaviour and evaluation results become unexplainable).
- sentence-transformers is imported lazily: the ingest/quant paths must not
  require torch just to run.
"""

from __future__ import annotations

from typing import Any, Iterable

from app.config import Config


class Embedder:
    """Thin wrapper around a sentence-transformers model."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        version: str | None = None,
        dim: int | None = None,
    ) -> None:
        self.model_name = model_name or Config.EMBEDDING_MODEL
        self.device = device or Config.EMBEDDING_DEVICE
        self.batch_size = batch_size or Config.EMBEDDING_BATCH_SIZE
        self.version = version or Config.EMBEDDING_VERSION
        self.dim = dim or Config.EMBEDDING_DIM
        self._model: Any = None
        self._loaded_device = ""

    # -- model lifecycle ---------------------------------------------------- #
    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError(
                "sentence-transformers is required for embeddings; install it with: "
                "pip install sentence-transformers"
            ) from exc

        device = self.device
        if device and device != "auto":
            try:
                self._model = SentenceTransformer(self.model_name, device=device)
                self._loaded_device = device
                return self._model
            except Exception as exc:  # noqa: BLE001 - fall back to CPU loudly
                print(
                    f"[warn] could not load {self.model_name} on device={device} "
                    f"({type(exc).__name__}: {exc}); falling back to cpu"
                )
        self._model = SentenceTransformer(self.model_name)
        self._loaded_device = "cpu(default)"
        return self._model

    @property
    def loaded_device(self) -> str:
        return self._loaded_device

    # -- inference ---------------------------------------------------------- #
    def encode(self, texts: Iterable[str]) -> list[list[float]]:
        """Embed texts; vectors are L2-normalized so cosine == dot product."""
        batch = list(texts)
        if not batch:
            return []
        model = self._load()
        vectors = model.encode(
            batch,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [list(map(float, vector)) for vector in vectors]

    def encode_one(self, text: str) -> list[float]:
        vectors = self.encode([text])
        return vectors[0]
