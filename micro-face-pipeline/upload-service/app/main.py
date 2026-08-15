from fastapi import FastAPI
from app.api.upload import router as upload_router
from app.api.health import router as health_router
from app.core.logger import logger

app = FastAPI(
    title="Photo Upload Service",
    description="Micro Face Clustering Upload API",
    version="1.0.0",
)

app.include_router(health_router)
app.include_router(upload_router)

@app.on_event("startup")
async def startup_event():
    logger.info("Photo Upload Service started.")


@app.get("/", tags=["Root"])
async def root():
    return {
        "service": "Photo Upload API",
        "project": "test-face-clustering",
        "status": "running"
    }