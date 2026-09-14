from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import (
    Face,
    FaceEmbedding,
    Image,
    ProcessingJob,
    Upload,
    Workflow1OutboxEvent,
)


logger = logging.getLogger(__name__)


WORKFLOW_NAME = "workflow1"


# ======================================================================
# UPLOAD REPOSITORY
# ======================================================================

class UploadRepository:

    def __init__(
        self,
        db: Session,
    ):
        self.db = db

    def get_by_event_id(
        self,
        event_id: UUID,
    ) -> Upload | None:

        return self.db.scalar(
            select(Upload).where(
                Upload.event_id == event_id
            )
        )

    def get_or_create_upload(
        self,
        *,
        event_id: UUID,
        event_version: str,
        event_type: str,
        project_id: str,
        bucket: str,
        blob_name: str,
        original_filename: str | None = None,
        content_type: str | None = None,
        file_size: int | None = None,
        uploaded_at: datetime | None = None,
        created_by: str | None = None,
    ) -> Upload:

        existing = self.get_by_event_id(
            event_id
        )

        if existing:

            return existing

        upload = Upload(
            event_id=event_id,
            event_version=event_version,
            event_type=event_type,
            project_id=project_id,
            bucket=bucket,
            blob_name=blob_name,
            original_filename=original_filename,
            content_type=content_type,
            file_size=file_size,
            uploaded_at=uploaded_at,
            created_by=created_by,
        )

        self.db.add(
            upload
        )

        self.db.flush()

        return upload


# ======================================================================
# PROCESSING JOB REPOSITORY
# ======================================================================

class ProcessingJobRepository:

    def __init__(
        self,
        db: Session,
    ):
        self.db = db

    def get(
        self,
        upload_id: int,
        workflow: str = WORKFLOW_NAME,
    ) -> ProcessingJob | None:

        return self.db.scalar(
            select(ProcessingJob).where(
                ProcessingJob.upload_id == upload_id,
                ProcessingJob.workflow == workflow,
            )
        )

    def get_or_create(
        self,
        upload_id: int,
        workflow: str = WORKFLOW_NAME,
    ) -> ProcessingJob:

        job = self.get(
            upload_id,
            workflow,
        )

        if job:

            return job

        job = ProcessingJob(
            upload_id=upload_id,
            workflow=workflow,
            status="RECEIVED",
            attempt_count=0,
        )

        self.db.add(
            job
        )

        self.db.flush()

        return job

    def mark_started(
        self,
        job: ProcessingJob,
    ) -> None:

        job.status = "PROCESSING"

        job.attempt_count += 1

        job.started_at = (
            datetime.now(timezone.utc)
        )

        self.db.flush()

    def mark_completed(
        self,
        job: ProcessingJob,
    ) -> None:

        job.status = "COMPLETED"

        job.completed_at = (
            datetime.now(timezone.utc)
        )

        job.error_code = None
        job.error_message = None

        self.db.flush()

    def mark_failed(
        self,
        job: ProcessingJob,
        error_code: str,
        error_message: str,
    ) -> None:

        job.status = "FAILED"

        job.error_code = error_code

        job.error_message = (
            error_message[:10000]
        )

        self.db.flush()


# ======================================================================
# WORKFLOW1 OUTBOX REPOSITORY
# ======================================================================


class Workflow1OutboxRepository:

    def __init__(
        self,
        db: Session,
    ):
        self.db = db

    def create(
        self,
        *,
        event_id: str,
        event_type: str,
        aggregate_type: str,
        aggregate_id: str,
        payload: str,
    ) -> Workflow1OutboxEvent:

        event = Workflow1OutboxEvent(
            event_id=event_id,
            event_type=event_type,
            aggregate_type=aggregate_type,
            aggregate_id=aggregate_id,
            payload=payload,
            status="PENDING",
            attempt_count=0,
        )

        self.db.add(event)

        self.db.flush()

        return event

    def get_pending(
        self,
        limit: int = 100,
    ) -> list[Workflow1OutboxEvent]:

        return list(
            self.db.scalars(
                select(Workflow1OutboxEvent)
                .where(
                    Workflow1OutboxEvent.status == "PENDING"
                )
                .order_by(
                    Workflow1OutboxEvent.created_at
                )
                .limit(limit)
            )
        )

    def mark_attempt(
        self,
        event: Workflow1OutboxEvent,
    ) -> None:

        event.attempt_count += 1

        self.db.flush()

    def mark_published(
        self,
        event: Workflow1OutboxEvent,
    ) -> None:

        event.status = "PUBLISHED"

        event.published_at = (
            datetime.now(timezone.utc)
        )

        event.last_error = None

        self.db.flush()

    def mark_failed(
        self,
        event: Workflow1OutboxEvent,
        error_message: str,
    ) -> None:

        event.status = "FAILED"

        event.last_error = error_message[:10000]

        self.db.flush()



# ======================================================================
# IMAGE REPOSITORY
# ======================================================================

class ImageRepository:

    def __init__(
        self,
        db: Session,
    ):
        self.db = db

    def get_by_object(
        self,
        bucket: str,
        blob_name: str,
    ) -> Image | None:

        return self.db.scalar(
            select(Image).where(
                Image.bucket == bucket,
                Image.blob_name == blob_name,
            )
        )

    def get_or_create(
        self,
        *,
        upload_id: int,
        bucket: str,
        blob_name: str,
        content_type: str | None = None,
        file_size: int | None = None,
    ) -> Image:

        existing = self.get_by_object(
            bucket,
            blob_name,
        )

        if existing:

            return existing

        image = Image(
            upload_id=upload_id,
            bucket=bucket,
            blob_name=blob_name,
            content_type=content_type,
            file_size=file_size,
        )

        self.db.add(
            image
        )

        self.db.flush()

        return image

    def mark_downloaded(
        self,
        image: Image,
        *,
        local_path: str,
        width: int | None = None,
        height: int | None = None,
        channels: int | None = None,
        image_hash: str | None = None,
    ) -> None:

        image.local_path = local_path
        image.width = width
        image.height = height
        image.channels = channels
        image.image_hash = image_hash

        image.downloaded_at = (
            datetime.now(timezone.utc)
        )

        self.db.flush()


# ======================================================================
# FACE REPOSITORY
# ======================================================================

class FaceRepository:

    def __init__(
        self,
        db: Session,
    ):
        self.db = db

    def get_by_image_and_index(
        self,
        image_id: int,
        face_index: int,
    ) -> Face | None:

        return self.db.scalar(
            select(Face).where(
                Face.image_id == image_id,
                Face.face_index == face_index,
            )
        )

    def create_or_update(
        self,
        *,
        image_id: int,
        face_index: int,
        detection_confidence: float,
        bbox: list[float],
        gender: str | None = None,
        gender_confidence: float | None = None,
        age_years: int | None = None,
        age_group: str | None = None,
        age_confidence: float | None = None,
        landmarks: list[float] | None = None,
    ) -> Face:

        if len(bbox) != 4:

            raise ValueError(
                "Face bbox must contain exactly 4 values."
            )

        if landmarks is not None:

            if len(landmarks) != 10:

                raise ValueError(
                    "Face landmarks must contain exactly 10 values."
                )

        face = self.get_by_image_and_index(
            image_id,
            face_index,
        )

        values = {
            "detection_confidence": float(
                detection_confidence
            ),

            "bbox_x1": float(
                bbox[0]
            ),

            "bbox_y1": float(
                bbox[1]
            ),

            "bbox_x2": float(
                bbox[2]
            ),

            "bbox_y2": float(
                bbox[3]
            ),

            "gender": gender,

            "gender_confidence": (
                float(gender_confidence)
                if gender_confidence is not None
                else None
            ),

            "age_years": (
                int(age_years)
                if age_years is not None
                else None
            ),

            "age_group": age_group,

            "age_confidence": (
                float(age_confidence)
                if age_confidence is not None
                else None
            ),
        }

        if landmarks is not None:

            for i in range(5):

                values[
                    f"landmark_{i + 1}_x"
                ] = float(
                    landmarks[i * 2]
                )

                values[
                    f"landmark_{i + 1}_y"
                ] = float(
                    landmarks[i * 2 + 1]
                )

        if face is None:

            face = Face(
                image_id=image_id,
                face_index=face_index,
                **values,
            )

            self.db.add(
                face
            )

        else:

            for key, value in values.items():

                setattr(
                    face,
                    key,
                    value,
                )

        self.db.flush()

        return face


# ======================================================================
# FACE EMBEDDING REPOSITORY
# ======================================================================

class FaceEmbeddingRepository:

    def __init__(
        self,
        db: Session,
    ):
        self.db = db

    def get_by_face_id(
        self,
        face_id: int,
    ) -> FaceEmbedding | None:

        return self.db.scalar(
            select(FaceEmbedding).where(
                FaceEmbedding.face_id == face_id
            )
        )

    def create_or_update(
        self,
        *,
        face_id: int,
        model_name: str,
        model_version: str | None,
        embedding_dimension: int,
        vector_store: str,
        vector_id: str,
        normalized: bool = True,
        embedding_norm: float | None = None,
    ) -> FaceEmbedding:

        embedding = self.get_by_face_id(
            face_id
        )

        values = {
            "model_name": model_name,

            "model_version": model_version,

            "embedding_dimension": int(
                embedding_dimension
            ),

            "vector_store": vector_store,

            "vector_id": vector_id,

            "normalized": bool(
                normalized
            ),

            "embedding_norm": (
                float(embedding_norm)
                if embedding_norm is not None
                else None
            ),
        }

        if embedding is None:

            embedding = FaceEmbedding(
                face_id=face_id,
                **values,
            )

            self.db.add(
                embedding
            )

        else:

            for key, value in values.items():

                setattr(
                    embedding,
                    key,
                    value,
                )

        self.db.flush()

        return embedding