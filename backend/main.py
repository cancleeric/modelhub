from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import submissions, registry, actions
from models import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="ModelHub", version="0.2.0", lifespan=lifespan)

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


@app.get("/health")
def health():
    return {"status": "ok", "service": "modelhub", "version": "0.2.0"}
