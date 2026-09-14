import uuid
from datetime import datetime, timezone
from pathlib import Path

from shared.constants import (
    EVENT_VERSION,
    PROJECT_ID,
)

from shared.models.upload_event import UploadEvent


class UploadEventService:
    """
    Converts a Google Cloud Storage event into
    a standardized UploadEvent.
    """

    def create_event(self, storage_event: dict) -> UploadEvent:

        blob_name = storage_event["name"]

        return UploadEvent(
            event_id=str(uuid.uuid4()),
            event_version=EVENT_VERSION,
            event_type="UPLOAD_CREATED",
            project_id=PROJECT_ID,
            bucket=storage_event["bucket"],
            blob_name=blob_name,
            original_filename=Path(blob_name).name,
            content_type=storage_event.get(
                "contentType",
                "application/octet-stream",
            ),
            file_size=int(storage_event.get("size", 0)),
            uploaded_at=datetime.now(timezone.utc),
        )


upload_event_service = UploadEventService()