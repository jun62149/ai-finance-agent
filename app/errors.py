from __future__ import annotations


class QuantError(Exception):
    """Base class for all quant engine errors."""

    code = "QuantError"
    http_status = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidStockCode(QuantError):
    code = "InvalidStockCode"
    http_status = 400


class StockNotFound(QuantError):
    code = "StockNotFound"
    http_status = 404


class NoData(QuantError):
    code = "NoData"
    http_status = 404


class HistoricalPredictionUnavailable(QuantError):
    code = "HistoricalPredictionUnavailable"
    http_status = 501


class ModelUnavailable(QuantError):
    code = "ModelUnavailable"
    http_status = 503


class DatabaseError(QuantError):
    code = "DatabaseError"
    http_status = 500


class IndexOutOfSync(QuantError):
    """The BM25 index no longer matches the chunk corpus.

    Raised instead of silently answering from a stale index: a drifted index can
    return chunks that no longer exist (or, worse, hydrate someone else's text).
    """

    code = "IndexOutOfSync"
    http_status = 503
