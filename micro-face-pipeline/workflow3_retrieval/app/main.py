from __future__ import annotations

from fastapi import FastAPI

from app.api.dashboard import router as dashboard_router


app = FastAPI(
    title="Workflow 3 Retrieval API",
    version="1.0.0",
)


app.include_router(dashboard_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}