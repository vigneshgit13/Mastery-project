from fastapi import APIRouter, File, UploadFile

from app.models.upload_models import UploadResponse
from app.services.upload_service import upload_service

router = APIRouter(
    prefix="/upload",
    tags=["Upload"],
)


@router.post(
    "",
    response_model=UploadResponse,
)
async def upload_image(
    file: UploadFile = File(...),
):
    return await upload_service.upload(file)