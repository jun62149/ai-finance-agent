from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api.chat import router as chat_router
from app.api.rag import router as rag_router
from app.api.stocks import router
from app.errors import QuantError

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"


def create_app() -> FastAPI:
    application = FastAPI(
        title="ai-finance-agent Quant API",
        description=(
            "Quant Engine contract + hybrid RAG retrieval. Agent modules land in a "
            "later phase; this API never returns demo/fake data."
        ),
        version=__version__,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.exception_handler(QuantError)
    async def quant_error_handler(_: Request, exc: QuantError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.http_status,
            content={"detail": {"code": exc.code, "message": exc.message}},
        )

    application.include_router(router)
    application.include_router(rag_router)
    application.include_router(chat_router)

    # Minimal Web UI (P8): served by the same app, so one process serves both.
    if FRONTEND_DIR.exists():
        application.mount(
            "/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static"
        )

        @application.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(FRONTEND_DIR / "index.html")

    return application


app = create_app()
