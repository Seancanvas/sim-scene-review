from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from .api import ApiError, router
from .runs import Registry

DEFAULT_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Every open SSE client parks a thread in wait_for and the pipeline runs in
    # the same pool, so the default pool (cpu+4) would stall runs under a
    # handful of viewers.
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=64))
    yield


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def create_app() -> FastAPI:
    app = FastAPI(title="Sim-Scene Review", docs_url="/api/docs", openapi_url="/api/openapi.json", lifespan=_lifespan)
    app.state.registry = Registry()
    app.state.tasks = set()
    app.include_router(router)

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError):
        return _error(exc.status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        return _error(422, "validation_error", str(exc))

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        # unknown paths, 405s and bad path params otherwise answer with Starlette's {"detail": ...}
        return _error(exc.status_code, f"http_{exc.status_code}", str(exc.detail))

    dist = Path(os.environ.get("FRONTEND_DIST", str(DEFAULT_DIST)))
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="frontend")
    return app


app = create_app()
