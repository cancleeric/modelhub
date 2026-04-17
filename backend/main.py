import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import submissions, registry, actions, kaggle, api_keys, predict
from models import init_db
from pollers.kaggle_poller import start_scheduler, stop_scheduler
from version import VERSION, BUILD_INFO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    scheduler = start_scheduler()
    try:
        yield
    finally:
        if scheduler:
            stop_scheduler()


app = FastAPI(title="ModelHub", version=VERSION, lifespan=lifespan)

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


@app.get("/health")
def health():
    return {"status": "ok", "service": "modelhub", **BUILD_INFO}


@app.get("/version")
def version():
    return BUILD_INFO
