from __future__ import annotations
from app.api.media import router as media_router
from app.core.config import CORS_ORIGINS
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.dashboard import router as dashboard_router

app = FastAPI(
    title="Workflow 3 Retrieval API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dashboard_router)
app.include_router(media_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}