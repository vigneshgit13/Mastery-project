from __future__ import annotations

from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class FaceExtractionCompletedEvent(BaseModel):
    """
    Workflow 1 -> Workflow 2 completion event.

    This event means Workflow 1 has successfully persisted
    the faces and embeddings for one image.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    event_version: str = "1.0"

    event_type: str = "FACE_EXTRACTION_COMPLETED"

    event_id: UUID

    upload_id: int

    image_id: int

    bucket: str

    blob_name: str

    face_count: int = Field(
        ge=0
    )

    def validate_event_type(self) -> None:
        if self.event_type != "FACE_EXTRACTION_COMPLETED":
            raise ValueError(
                "Unexpected event_type: "
                f"{self.event_type}"
            )

    @classmethod
    def create(
        cls,
        *,
        upload_id: int,
        image_id: int,
        bucket: str,
        blob_name: str,
        face_count: int,
    ) -> "FaceExtractionCompletedEvent":

        return cls(
            event_version="1.0",
            event_type="FACE_EXTRACTION_COMPLETED",
            event_id=uuid4(),
            upload_id=upload_id,
            image_id=image_id,
            bucket=bucket,
            blob_name=blob_name,
            face_count=face_count,
        )