from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class UploadEvent(BaseModel):

    model_config = ConfigDict(
        extra="ignore"
    )

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

    @classmethod
    def from_dict(
        cls,
        payload: dict,
    ) -> "UploadEvent":

        return cls.model_validate(
            payload
        )