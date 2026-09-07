from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api.stocks import router
from app.errors import QuantError


def create_app() -> FastAPI:
    application = FastAPI(
        title="ai-finance-agent Quant API",
        description=(
            "Phase 0 Quant Engine contract. RAG/Agent modules land in later phases; "
            "this API never returns demo/fake data."
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
    return application


app = create_app()
