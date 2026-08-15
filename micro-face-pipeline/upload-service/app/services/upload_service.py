import os
import tempfile

from fastapi import UploadFile

from app.models.upload_models import (
    UploadMetadata,
    UploadResponse,
)
from app.services.gcs_service import gcs_service
from app.utils.file_validator import FileValidator


class UploadService:

    async def upload(
        self,
        file: UploadFile,
    ) -> UploadResponse:

        temp_path = None

        try:

            suffix = os.path.splitext(file.filename)[1]

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as temp_file:

                content = await file.read()

                temp_file.write(content)

                temp_path = temp_file.name

            metadata = UploadMetadata(
                original_filename=file.filename,
                content_type=file.content_type,
                file_size=os.path.getsize(temp_path),
            )

            FileValidator.validate(
                metadata.original_filename,
                metadata.file_size,
            )

            return gcs_service.upload_file(
                file_path=temp_path,
                metadata=metadata,
            )
        finally:

            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


upload_service = UploadService()