import uuid
from datetime import datetime, timezone
from pathlib import Path

from google.cloud import storage

from app.core.config import settings
from app.core.logger import logger
from app.models.upload_models import (
    UploadMetadata,
    UploadResponse,
)


class GCSService:
    def __init__(self):
        self.client = storage.Client(project=settings.PROJECT_ID)
        self.bucket = self.client.bucket(settings.BUCKET_NAME)

    def upload_file(
    self,
    file_path: str,
    metadata: UploadMetadata,
) -> UploadResponse:
        """
        Upload a file to Google Cloud Storage.

        Returns metadata about the uploaded file.
        """

        upload_id = str(uuid.uuid4())

        extension = Path(metadata.original_filename).suffix

        blob_name = f"uploads/{upload_id}{extension}"

        blob = self.bucket.blob(blob_name)

        blob.upload_from_filename(
            filename=file_path,
            content_type=metadata.content_type,
        )

        logger.info(
            f"Uploaded {metadata.original_filename} -> {blob_name}"
        )

        return UploadResponse(
    upload_id=upload_id,
    original_filename=metadata.original_filename,
    content_type=metadata.content_type,
    file_size=metadata.file_size,
    bucket=settings.BUCKET_NAME,
    blob_name=blob_name,
    uploaded_at=datetime.now(timezone.utc).isoformat(),
    status="SUCCESS",
)

gcs_service = GCSService()