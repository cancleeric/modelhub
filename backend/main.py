from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import submissions, registry

app = FastAPI(title="ModelHub", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(submissions.router, prefix="/api/submissions", tags=["submissions"])
app.include_router(registry.router, prefix="/api/registry", tags=["registry"])


@app.get("/health")
def health():
    return {"status": "ok", "service": "modelhub"}
