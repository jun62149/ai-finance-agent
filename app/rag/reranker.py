"""Cross-encoder reranker (Phase 3).

The reranker only ever scores the candidate set produced by recall (dense +
BM25 + RRF), never the whole corpus — that is what keeps it affordable.

It is optional by design:
- `RERANKER_ENABLED=0` disables it;
- when the model cannot be loaded the pipeline degrades to fusion order and the
  caller is told (`reranked=False` plus a `rerank_error`), instead of failing.
"""

from __future__ import annotations

from typing import Any, Sequence

from app.config import Config


class Reranker:
    """Thin wrapper around a sentence-transformers CrossEncoder."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        enabled: bool | None = None,
        batch_size: int = 16,
        local_files_only: str | None = None,
    ) -> None:
        self.model_name = model_name or Config.RERANKER_MODEL
        self.device = device or Config.RERANKER_DEVICE
        self.enabled = Config.RERANKER_ENABLED if enabled is None else enabled
        self.batch_size = batch_size
        self.local_files_only = str(
            local_files_only or Config.EMBEDDING_LOCAL_FILES_ONLY
        ).lower()
        self._model: Any = None
        self._load_error: str = ""

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
        devices.append(None)
        return [(flag, device) for flag in flags for device in devices]

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError(
                "sentence-transformers is required for reranking; install it with: "
                "pip install sentence-transformers"
            ) from exc

        errors: list[str] = []
        for local_only, device in self._strategies():
            try:
                self._model = CrossEncoder(
                    self.model_name,
                    device=device,
                    local_files_only=local_only,
                    max_length=512,
                )
                return self._model
            except Exception as exc:  # noqa: BLE001 - try the next strategy
                errors.append(
                    f"local_files_only={local_only} device={device}: "
                    f"{type(exc).__name__}: {exc}"
                )
        raise RuntimeError(
            f"could not load reranker {self.model_name!r}. Tried: " + " | ".join(errors)
        )

    # -- scoring ------------------------------------------------------------ #
    def rerank(
        self,
        query: str,
        candidates: Sequence[dict[str, Any]],
        *,
        top_k: int | None = None,
        content_key: str = "content",
    ) -> tuple[list[dict[str, Any]], str]:
        """Score query/candidate pairs. Returns (results, error_message).

        On failure the candidates are returned in their original order and the
        error message is non-empty — callers must surface that degradation.
        """
        candidates = list(candidates)
        if not candidates:
            return [], ""
        if not self.enabled:
            return candidates[:top_k] if top_k else candidates, "disabled"

        try:
            model = self._load()
            pairs = [
                (query, str(candidate.get(content_key) or "")) for candidate in candidates
            ]
            scores = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        except Exception as exc:  # noqa: BLE001 - degrade instead of failing the request
            return (
                candidates[:top_k] if top_k else candidates,
                f"{type(exc).__name__}: {exc}",
            )

        ranked = []
        for candidate, score in zip(candidates, scores):
            item = dict(candidate)
            item["rerank_score"] = float(score)
            ranked.append(item)
        ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        return (ranked[:top_k] if top_k else ranked), ""
