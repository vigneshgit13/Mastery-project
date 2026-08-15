from datetime import datetime

from pydantic import BaseModel


class UploadEvent(BaseModel):

    event_id: str

    event_version: str

    event_type: str

    project_id: str

    bucket: str

    blob_name: str

    original_filename: str

    content_type: str

    file_size: int

    uploaded_at: datetime

    created_by: str