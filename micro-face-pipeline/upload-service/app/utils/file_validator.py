from pathlib import Path

from fastapi import HTTPException

from app.core.config import settings


class FileValidator:

    @staticmethod
    def validate(filename: str, file_size: int):

        extension = Path(filename).suffix.lower()

        if extension not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {extension}",
            )

        max_size = settings.MAX_FILE_SIZE_MB * 1024 * 1024

        if file_size > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"Maximum file size is {settings.MAX_FILE_SIZE_MB} MB",
            )