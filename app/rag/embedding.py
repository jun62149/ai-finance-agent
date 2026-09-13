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
    """Thin wrapper around a sentence-transformers model.

    Loading strategy (`EMBEDDING_LOCAL_FILES_ONLY` = auto | 1 | 0):
    - auto (default): try the local HuggingFace cache first, then the network.
      This matters because a cached model still triggers remote HEAD checks
      (and those hang/time out on networks where huggingface.co is blocked).
    - 1: offline only (fail fast when the model is not cached).
    - 0: network allowed first.
    Device fallback (cuda -> default) is retried and reported loudly.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        version: str | None = None,
        dim: int | None = None,
        local_files_only: str | None = None,
    ) -> None:
        self.model_name = model_name or Config.EMBEDDING_MODEL
        self.device = device or Config.EMBEDDING_DEVICE
        self.batch_size = batch_size or Config.EMBEDDING_BATCH_SIZE
        self.version = version or Config.EMBEDDING_VERSION
        self.dim = dim or Config.EMBEDDING_DIM
        self.local_files_only = str(
            local_files_only or Config.EMBEDDING_LOCAL_FILES_ONLY
        ).lower()
        self._model: Any = None
        self._loaded_device = ""
        self._loaded_local_only = False

    # -- model lifecycle ---------------------------------------------------- #
    def _strategies(self) -> list[tuple[bool, str | None]]:
        if self.local_files_only == "1":
            flags = [True]
        elif self.local_files_only == "0":
            flags = [False]
        else:
            flags = [True, False]
        devices: list[str | None] = []
        if self.device and self.device != "auto":
            devices.append(self.device)
        devices.append(None)  # let sentence-transformers pick (cpu)
        return [(flag, device) for flag in flags for device in devices]

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

        errors: list[str] = []
        for local_only, device in self._strategies():
            try:
                model = SentenceTransformer(
                    self.model_name, device=device, local_files_only=local_only
                )
            except Exception as exc:  # noqa: BLE001 - try the next strategy
                errors.append(
                    f"local_files_only={local_only} device={device}: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            self._model = model
            self._loaded_device = device or "default"
            self._loaded_local_only = local_only
            if not local_only:
                print(
                    f"[info] loaded {self.model_name} from the Hub "
                    "(not in the local cache)"
                )
            elif errors:
                print(f"[warn] falling back after: {errors[-1]}")
            return self._model

        raise RuntimeError(
            f"could not load embedding model {self.model_name!r}. Tried: "
            + " | ".join(errors)
            + ". If the model must be downloaded, set HF_ENDPOINT="
            "https://hf-mirror.com (or pre-download it)."
        )

    @property
    def loaded_device(self) -> str:
        return self._loaded_device

    @property
    def loaded_local_only(self) -> bool:
        return self._loaded_local_only

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
