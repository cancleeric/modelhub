import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from routers import submissions, registry, actions, kaggle, api_keys, predict, health, inference, queue as queue_router
from routers import comments as comments_router
from routers import attachments as attachments_router
from routers import notifications as notifications_router
from routers import external_models as external_models_router
from models import init_db
from pollers.kaggle_poller import start_scheduler, stop_scheduler
import pollers.lightning_poller as _lightning_poller
import pollers.queue_dispatcher as _queue_dispatcher
import pollers.health_checker as _health_checker
import pollers.ssh_poller as _ssh_poller
from version import VERSION, BUILD_INFO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    scheduler = start_scheduler()
    lightning_scheduler = _lightning_poller.start_scheduler()
    queue_dispatcher_scheduler = _queue_dispatcher.start_scheduler()
    health_checker_scheduler = _health_checker.start_scheduler()
    ssh_poller_scheduler = _ssh_poller.start_scheduler()
    try:
        yield
    finally:
        if scheduler:
            stop_scheduler()
        if lightning_scheduler:
            _lightning_poller.stop_scheduler()
        if queue_dispatcher_scheduler:
            _queue_dispatcher.stop_scheduler()
        if health_checker_scheduler:
            _health_checker.stop_scheduler()
        if ssh_poller_scheduler:
            _ssh_poller.stop_scheduler()


app = FastAPI(title="ModelHub", version=VERSION, lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    msgs = "; ".join(
        f"{'.'.join(str(l) for l in e['loc'] if l != 'body')}: {e['msg']}"
        for e in exc.errors()
    )
    return JSONResponse(status_code=422, content={"detail": msgs})


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3950",
        "http://localhost:8950",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(submissions.router, prefix="/api/submissions", tags=["submissions"])
app.include_router(registry.router, prefix="/api/registry", tags=["registry"])
app.include_router(actions.router, prefix="/api/submissions", tags=["actions"])
app.include_router(kaggle.router, prefix="/api/submissions", tags=["kaggle"])
app.include_router(api_keys.router, prefix="/api/admin/api-keys", tags=["admin"])
app.include_router(predict.router, prefix="/api/predict", tags=["predict"])
app.include_router(health.router, prefix="/api/health", tags=["health"])
app.include_router(inference.router, prefix="/v1", tags=["inference"])
app.include_router(queue_router.router, prefix="/api/queue", tags=["queue"])
app.include_router(comments_router.router, prefix="/api", tags=["comments"])
app.include_router(attachments_router.router, prefix="/api", tags=["attachments"])
app.include_router(notifications_router.router, prefix="/api", tags=["notifications"])
app.include_router(external_models_router.router, prefix="/api/external-models", tags=["external-models"])


@app.get("/health")
def health():
    from pollers.kaggle_poller import get_last_poll_at, _scheduler as _sched
    last_poll = get_last_poll_at()
    return {
        "status": "ok",
        "service": "modelhub",
        "poller_last_run": last_poll.isoformat() if last_poll else None,
        "scheduler_running": bool(_sched and _sched.running),
        **BUILD_INFO,
    }


@app.get("/version")
def version():
    return BUILD_INFO


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title="ModelHub API",
        version=VERSION,
        routes=app.routes,
    )
    schema.setdefault("components", {})
    schema["components"]["securitySchemes"] = {
        "BearerAuth": {"type": "http", "scheme": "bearer"},
        "ApiKeyHeader": {"type": "apiKey", "in": "header", "name": "X-Api-Key"},
    }
    schema["security"] = [{"BearerAuth": []}, {"ApiKeyHeader": []}]
    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi  # type: ignore[method-assign]
