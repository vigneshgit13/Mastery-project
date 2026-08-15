from datetime import datetime
from pydantic import BaseModel, Field


class UploadEvent(BaseModel):
    """
    Event published when a new image is uploaded to Google Cloud Storage.
    """

    event_id: str

    event_version: str = Field(default="1.0")

    event_type: str = Field(default="UPLOAD_CREATED")

    project_id: str

    bucket: str

    blob_name: str

    original_filename: str

    content_type: str

    file_size: int

    uploaded_at: datetime

    created_by: str = Field(default="upload-service")