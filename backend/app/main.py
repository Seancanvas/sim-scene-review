from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
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
        # Only the asset folder is a mount; API paths must never fall into a
        # catch-all, or a wrong-method call would answer 404 instead of 405.
        if (dist / "assets").is_dir():
            app.mount("/assets", StaticFiles(directory=str(dist / "assets")), name="assets")

        @app.get("/", include_in_schema=False)
        async def _index():
            return FileResponse(dist / "index.html")

        @app.get("/{filename}", include_in_schema=False)
        async def _root_file(filename: str):
            # single-segment root files only (favicon, icons); two-segment paths like /api/x never match here
            target = dist / filename
            if target.is_file():
                return FileResponse(target)
            raise ApiError(404, "http_404", "Not Found")
    return app


app = create_app()
