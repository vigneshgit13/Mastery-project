import logging
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Upload, Image, Face


logger = logging.getLogger(__name__)


class UploadRepository:

    def __init__(self, db: Session):
        self.db = db

    def get_by_event_id(
        self,
        event_id: str,
    ) -> Upload | None:

        return self.db.scalar(
            select(Upload)
            .where(
                Upload.event_id == event_id
            )
        )

    def create_upload(
        self,
        *,
        event_id: str | None,
        bucket_name: str,
        blob_name: str,
        original_filename: str | None,
    ) -> Upload:

        existing = None

        if event_id:
            existing = self.get_by_event_id(
                event_id
            )

        if existing:
            logger.info(
                "Upload already exists: %s",
                existing.id,
            )

            return existing

        upload = Upload(
            event_id=event_id,
            bucket_name=bucket_name,
            blob_name=blob_name,
            original_filename=original_filename,
            status="RECEIVED",
        )

        self.db.add(upload)
        self.db.flush()

        return upload

    def update_status(
        self,
        upload: Upload,
        status: str,
        error_message: str | None = None,
    ) -> None:

        upload.status = status
        upload.error_message = error_message

        self.db.flush()


class FaceRepository:

    def __init__(self, db: Session):
        self.db = db

    def create_image(
        self,
        *,
        upload_id: uuid.UUID,
        gcs_uri: str,
        width: int | None = None,
        height: int | None = None,
    ) -> Image:

        image = Image(
            upload_id=upload_id,
            gcs_uri=gcs_uri,
            width=width,
            height=height,
            status="PROCESSING",
        )

        self.db.add(image)
        self.db.flush()

        return image

    def create_face(
        self,
        *,
        image_id: uuid.UUID,
        face_index: int,
        record: dict[str, Any],
    ) -> Face:

        detection_confidence = float(
            record.get(
                "detection_confidence",
                0.0,
            )
        )

        gender = record.get(
            "gender"
        )

        gender_confidence = record.get(
            "gender_confidence"
        )

        age_years = record.get(
            "age_years"
        )

        age_group = record.get(
            "age_group"
        )

        age_confidence = record.get(
            "age_confidence"
        )

        bbox = record.get(
            "bbox",
            {},
        )

        embedding_dimension = record.get(
            "embedding_dimension"
        )

        embedding_id = record.get(
            "embedding_id"
        )

        face = Face(
            image_id=image_id,
            face_index=face_index,
            bbox=bbox,
            detection_confidence=detection_confidence,
            gender=gender,
            gender_confidence=(
                float(gender_confidence)
                if gender_confidence is not None
                else None
            ),
            age_years=(
                float(age_years)
                if age_years is not None
                else None
            ),
            age_group=age_group,
            age_confidence=(
                float(age_confidence)
                if age_confidence is not None
                else None
            ),
            embedding_id=embedding_id,
            embedding_dimension=(
                int(embedding_dimension)
                if embedding_dimension is not None
                else None
            ),
        )

        self.db.add(face)

        return face