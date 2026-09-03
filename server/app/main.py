"""App factory: load artifacts once (lifespan), map domain errors, mount routes, CORS."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import routes, services
from app.config import Settings
from app.state import BadInput, NotFound, NotReady


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.oe = services.build_state(app.state.settings)   # warm the artifacts
    yield
    try:
        app.state.oe.store.close()
    except Exception:                       # noqa: BLE001 — nothing to close is fine
        pass


def create_app() -> FastAPI:
    settings = Settings()
    app = FastAPI(title="OceanEmbed API", version="1.0", lifespan=lifespan)
    app.state.settings = settings

    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_list(),
                       allow_methods=["GET"], allow_headers=["*"])

    for exc, code in ((NotReady, 503), (NotFound, 404), (BadInput, 422)):
        app.add_exception_handler(exc, _handler(code))

    app.include_router(routes.router)

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz(request: Request):
        r = services.readiness(request.app.state.oe)
        return JSONResponse(r, status_code=200 if r["ready"] else 503)

    return app


def _handler(code):
    async def h(request: Request, exc: Exception):
        return JSONResponse({"detail": str(exc)}, status_code=code)
    return h


app = create_app()
