from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any
import json
import cv2
from uuid import uuid4
import numpy as np

from app.models.face_extraction_completed import(
    FaceExtractionCompletedEvent,
)

from app.core.config import DOWNLOAD_DIR
from app.db.postgres import SessionLocal
from app.db.repository import (
    FaceEmbeddingRepository,
    FaceRepository,
    ImageRepository,
    ProcessingJobRepository,
    UploadRepository,
    Workflow1OutboxRepository,
)
from app.models.upload_event import UploadEvent
from app.services.faiss_service import FaissService
from app.services.gcs_service import GCSService
from app.services.redis_service import RedisService


logger = logging.getLogger(__name__)


# ============================================================
# WORKFLOW CONFIGURATION
# ============================================================

WORKFLOW_NAME = "workflow1"

VECTOR_STORE_NAME = "hnsw"

EMBEDDING_MODEL_NAME = "arcface"

EMBEDDING_MODEL_VERSION = "glintr100"

EMBEDDING_DIMENSION = 512

REDIS_STATE_TTL = 300


class ProcessingService:
    """
    Main business orchestration layer for Workflow 1.

    Workflow 1:

        UploadEvent
            |
            v
        PostgreSQL upload
            |
            v
        processing_job
            |
            v
        GCS download
            |
            v
        images
            |
            v
        FacePipeline
            |
            +----------------------+
            |                      |
            v                      v
        PostgreSQL              HNSW
        faces                    vector
            |                      |
            v                      v
        face_embeddings       Redis state
            |
            v
        COMPLETED

    Responsibilities:

    - persist upload state
    - create/update processing job
    - download image from GCS
    - register image metadata
    - execute the existing FacePipeline
    - persist detected faces
    - persist embedding metadata
    - insert embeddings into HNSW/FAISS
    - update Redis operational state
    - handle failures
    - maintain idempotency

    This class does NOT perform:

    - face detection
    - face alignment
    - face recognition
    - gender classification
    - age classification
    - Pub/Sub acknowledgement

    Those responsibilities remain in their dedicated components.
    """

    def __init__(
        self,
        *,
        pipeline,
        gcs_service: GCSService | None = None,
        faiss_service: FaissService | None = None,
        redis_service: RedisService | None = None,
    ) -> None:

        self.pipeline = pipeline

        self.gcs_service = (
            gcs_service
            if gcs_service is not None
            else GCSService()
        )

        self.faiss_service = (
            faiss_service
            if faiss_service is not None
            else FaissService()
        )

        self.redis_service = (
            redis_service
            if redis_service is not None
            else RedisService()
        )

        logger.info(
            "ProcessingService initialized."
        )

    # ============================================================
    # PUBLIC ENTRY POINT
    # ============================================================

    def process_upload(
        self,
        event: UploadEvent,
    ) -> dict[str, Any]:
        """
        Process one upload event.

        The Pub/Sub subscriber should ACK the message only after
        this method returns successfully.

        Any exception is deliberately re-raised so the subscriber
        can NACK the Pub/Sub message.
        """

        logger.info(
            "============================================================"
        )
        logger.info(
            "WORKFLOW 1 START"
        )
        logger.info(
            "event_id=%s",
            event.event_id,
        )
        logger.info(
            "bucket=%s",
            event.bucket,
        )
        logger.info(
            "blob=%s",
            event.blob_name,
        )
        logger.info(
            "============================================================"
        )

        db = SessionLocal()

        upload = None
        job = None

        try:

            # ========================================================
            # 1. UPLOAD RECORD
            # ========================================================

            upload_repo = UploadRepository(db)

            upload = upload_repo.get_or_create_upload(
                event_id=event.event_id,
                event_version=event.event_version,
                event_type=event.event_type,
                project_id=event.project_id,
                bucket=event.bucket,
                blob_name=event.blob_name,
                original_filename=event.original_filename,
                content_type=event.content_type,
                file_size=event.file_size,
                uploaded_at=event.uploaded_at,
                created_by=event.created_by,
            )

            logger.info(
                "Upload record ready: upload_id=%s",
                upload.id,
            )

            # ========================================================
            # 2. PROCESSING JOB
            # ========================================================

            job_repo = ProcessingJobRepository(db)

            job = job_repo.get_or_create(
                upload_id=upload.id,
                workflow=WORKFLOW_NAME,
            )

            logger.info(
                "Processing job ready: job_id=%s status=%s",
                job.id,
                job.status,
            )

            # ========================================================
            # 3. IDEMPOTENCY
            # ========================================================

            if job.status == "COMPLETED":

                logger.info(
                    "Event already completed. "
                    "Skipping duplicate processing: event_id=%s",
                    event.event_id,
                )

                db.commit()

                self._set_upload_state(
                    event=event,
                    status="ALREADY_COMPLETED",
                    upload_id=upload.id,
                    extra={
                        "job_id": job.id,
                    },
                )

                return {
                    "event_id": str(event.event_id),
                    "upload_id": upload.id,
                    "job_id": job.id,
                    "status": "ALREADY_COMPLETED",
                    "face_count": 0,
                }

            # ========================================================
            # 4. MARK JOB STARTED
            # ========================================================

            job_repo.mark_started(job)

            db.commit()

            self._set_upload_state(
                event=event,
                status="PROCESSING",
                upload_id=upload.id,
                extra={
                    "job_id": job.id,
                },
            )

            # ========================================================
            # 5. DOWNLOAD IMAGE FROM GCS
            # ========================================================

            local_path = self._download_image(
                event=event,
            )

            logger.info(
                "GCS download complete: %s",
                local_path,
            )

            # ========================================================
            # 6. CREATE IMAGE RECORD
            # ========================================================

            image_repo = ImageRepository(db)

            image = image_repo.get_or_create(
                upload_id=upload.id,
                bucket=event.bucket,
                blob_name=event.blob_name,
                content_type=event.content_type,
                file_size=event.file_size,
            )

            logger.info(
                "Image record ready: image_id=%s",
                image.id,
            )

            # ========================================================
            # 7. INSPECT IMAGE
            # ========================================================

            image_info = self._inspect_image(
                local_path,
            )

            image_repo.mark_downloaded(
                image,
                local_path=str(local_path),
                width=image_info["width"],
                height=image_info["height"],
                channels=image_info["channels"],
                image_hash=image_info["image_hash"],
            )

            db.commit()

            # ========================================================
            # 8. RUN EXISTING FACE PIPELINE
            # ========================================================

            logger.info(
                "Running FacePipeline..."
            )

            pipeline_result = self._run_pipeline(
                local_path,
            )

            faces = self._extract_faces(
                pipeline_result,
            )

            logger.info(
                "FacePipeline detected %d face(s).",
                len(faces),
            )

            # ========================================================
            # 9. PERSIST FACE RESULTS
            # ========================================================

            face_repo = FaceRepository(db)

            embedding_repo = FaceEmbeddingRepository(db)

            persisted_faces: list[dict[str, Any]] = []

            for face_position, face_data in enumerate(faces):

                logger.info(
                    "Persisting face %d/%d",
                    face_position + 1,
                    len(faces),
                )

                face = self._persist_face(
                    face_repo=face_repo,
                    image_id=image.id,
                    face_index=face_position,
                    face_data=face_data,
                )

                # ----------------------------------------------------
                # Embedding
                # ----------------------------------------------------

                embedding = self._extract_embedding(
                    face_data,
                )

                embedding_vector = np.asarray(
                    embedding,
                    dtype=np.float32,
                ).reshape(-1)

                if embedding_vector.size != EMBEDDING_DIMENSION:

                    raise ValueError(
                        "Invalid ArcFace embedding dimension: "
                        f"received={embedding_vector.size}, "
                        f"expected={EMBEDDING_DIMENSION}"
                    )

                vector_id = f"face:{face.id}"

                # ----------------------------------------------------
                # HNSW
                # ----------------------------------------------------

                vector_result = self.faiss_service.add(
                    embedding=embedding_vector,
                    vector_id=vector_id,
                    metadata={
                        "face_id": face.id,
                        "image_id": image.id,
                        "upload_id": upload.id,
                    },
                    persist=False,
                )

                # ----------------------------------------------------
                # PostgreSQL embedding metadata
                # ----------------------------------------------------

                embedding_repo.create_or_update(
                    face_id=face.id,
                    model_name=EMBEDDING_MODEL_NAME,
                    model_version=EMBEDDING_MODEL_VERSION,
                    embedding_dimension=EMBEDDING_DIMENSION,
                    vector_store=VECTOR_STORE_NAME,
                    vector_id=vector_id,
                    normalized=True,
                    embedding_norm=float(
                        np.linalg.norm(
                            embedding_vector
                        )
                    ),
                )

                # ----------------------------------------------------
                # Redis face metadata
                # ----------------------------------------------------

                self._set_face_state(
                    face=face,
                    image_id=image.id,
                    upload_id=upload.id,
                    gender=self._extract_gender(
                        face_data
                    ),
                    age_group=self._extract_age_group(
                        face_data
                    ),
                )

                persisted_faces.append(
                    {
                        "face_id": face.id,
                        "face_index": face.face_index,
                        "vector_id": vector_id,
                        "faiss_id": vector_result["faiss_id"],
                        "added_to_vector_store": vector_result[
                            "added"
                        ],
                    }
                )

            # ========================================================
            # 10. SAVE HNSW INDEX
            # ========================================================

            logger.info(
                "Persisting HNSW index..."
            )

            self.faiss_service.save()

            logger.info(
                "HNSW persistence complete."
            )

           # ========================================================
              # 11. MARK WORKFLOW 1 COMPLETED + CREATE OUTBOX EVENT
           # ========================================================

            job_repo.mark_completed(
                job
            )

            # --------------------------------------------------------
              # Create the Workflow 1 -> Workflow 2 completion event.
            #
              # IMPORTANT:
            # This event is written to the SAME PostgreSQL
            # transaction as the processing completion state.
            #
            # Therefore:
            #
            #   processing_jobs = COMPLETED
            #   outbox event     = PENDING
            #
            # are committed atomically.
            # --------------------------------------------------------
            
            completion_event = FaceExtractionCompletedEvent.create(
                       upload_id=upload.id,
                       image_id=image.id,
                       bucket=event.bucket,
                       blob_name=event.blob_name,
                       face_count=len(
                       persisted_faces
                         ),
                       )
            outbox_repo = Workflow1OutboxRepository(
                         db
                       )

            outbox_repo.create(
                        event_id=str(
                        completion_event.event_id
                  ),
                        event_type=completion_event.event_type,
                        aggregate_type="image",
                        aggregate_id=str(
                        image.id
                 ),
                         payload=completion_event.model_dump_json(),
                    )
            logger.info(
                       "Workflow 1 completion event created in outbox: "
                       "event_id=%s image_id=%s face_count=%s",
                        completion_event.event_id,
                        image.id,
                        len(persisted_faces),
                    )

            # --------------------------------------------------------
               # COMMIT BOTH:
               #
               #   1. ProcessingJob = COMPLETED
               #   2. Outbox event   = PENDING
               #
               # --------------------------------------------------------


            db.commit()
            logger.info(
                       "PostgreSQL transaction committed: "
                       "processing completion + outbox event."
                    )
            # ========================================================
            # 12. REDIS COMPLETION STATE
            # ========================================================

            self._set_upload_state(
                event=event,
                status="COMPLETED",
                upload_id=upload.id,
                extra={
                    "job_id": job.id,
                    "image_id": image.id,
                    "face_count": len(
                        persisted_faces
                    ),
                },
            )

            # ========================================================
            # 13. RESULT
            # ========================================================

            result = {
                "event_id": str(event.event_id),
                "upload_id": upload.id,
                "job_id": job.id,
                "image_id": image.id,
                "status": "COMPLETED",
                "face_count": len(
                    persisted_faces
                ),
                "faces": persisted_faces,
            }

            logger.info(
                "============================================================"
            )
            logger.info(
                "WORKFLOW 1 COMPLETED"
            )
            logger.info(
                "event_id=%s faces=%d",
                event.event_id,
                len(persisted_faces),
            )
            logger.info(
                "============================================================"
            )

            return result

        except Exception as exc:

            logger.exception(
                "Workflow 1 failed: event_id=%s",
                event.event_id,
            )

            # --------------------------------------------------------
            # Roll back the active transaction.
            # --------------------------------------------------------

            db.rollback()

            # --------------------------------------------------------
            # Persist FAILED state in a fresh transaction.
            # --------------------------------------------------------

            if upload is not None:

                try:

                    failure_job_repo = (
                        ProcessingJobRepository(db)
                    )

                    failed_job = (
                        failure_job_repo.get_or_create(
                            upload_id=upload.id,
                            workflow=WORKFLOW_NAME,
                        )
                    )

                    failure_job_repo.mark_failed(
                        failed_job,
                        error_code=type(exc).__name__,
                        error_message=str(exc),
                    )

                    db.commit()

                except Exception:

                    logger.exception(
                        "Unable to persist FAILED job state."
                    )

                    db.rollback()

            # --------------------------------------------------------
            # Redis is auxiliary.
            # Redis failure must never hide the real exception.
            # --------------------------------------------------------

            self._set_upload_state(
                event=event,
                status="FAILED",
                upload_id=(
                    upload.id
                    if upload is not None
                    else None
                ),
                extra={
                    "error_code": type(exc).__name__,
                    "error": str(exc),
                },
            )

            # --------------------------------------------------------
            # VERY IMPORTANT:
            #
            # Re-raise so Pub/Sub subscriber NACKs the message.
            # --------------------------------------------------------

            raise

        finally:

            db.close()

    # ============================================================
    # GCS DOWNLOAD
    # ============================================================

    def _download_image(
        self,
        *,
        event: UploadEvent,
    ) -> Path:
        """
        Download the GCS object using the project's actual
        GCSService interface.
        """

        DOWNLOAD_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        suffix = Path(
            event.blob_name
        ).suffix

        if not suffix:
            suffix = ".jpg"

        destination = (
            DOWNLOAD_DIR
            / f"{event.event_id}{suffix}"
        )

        # --------------------------------------------------------
        # Your current GCSService downloads into DOWNLOAD_DIR
        # and returns the resulting local path.
        # --------------------------------------------------------

        result = self.gcs_service.download_file(
            bucket_name=event.bucket,
            blob_name=event.blob_name,
        )

        if result is not None:

            result_path = Path(
                str(result)
            )

            if result_path.exists():
                return result_path

        # --------------------------------------------------------
        # Fallback if the service returns nothing.
        # --------------------------------------------------------

        if destination.exists():
            return destination

        raise FileNotFoundError(
            "GCS download returned successfully, "
            "but the local image could not be found."
        )

    # ============================================================
    # IMAGE INSPECTION
    # ============================================================

    @staticmethod
    def _inspect_image(
        local_path: Path,
    ) -> dict[str, Any]:
        """
        Read image dimensions/channels and calculate SHA-256.
        """

        image = cv2.imread(
            str(local_path),
            cv2.IMREAD_UNCHANGED,
        )

        if image is None:

            raise ValueError(
                f"Unable to read downloaded image: "
                f"{local_path}"
            )

        height, width = image.shape[:2]

        if image.ndim == 2:
            channels = 1
        else:
            channels = image.shape[2]

        hasher = hashlib.sha256()

        with local_path.open(
            "rb"
        ) as file:

            for chunk in iter(
                lambda: file.read(
                    1024 * 1024
                ),
                b"",
            ):

                hasher.update(
                    chunk
                )

        return {
            "width": int(width),
            "height": int(height),
            "channels": int(channels),
            "image_hash": hasher.hexdigest(),
        }

    # ============================================================
    # FACE PIPELINE
    # ============================================================

    def _run_pipeline(
        self,
        local_path: Path,
    ) -> Any:
        """
        Execute the existing FacePipeline.

        The current pipeline contract is process(image_path).
        """

        if not hasattr(
            self.pipeline,
            "process",
        ):

            raise AttributeError(
                "FacePipeline does not expose "
                "the required process() method."
            )

        return self.pipeline.process(
            str(local_path)
        )

    # ============================================================
    # PIPELINE RESULT NORMALIZATION
    # ============================================================

    @staticmethod
    def _extract_faces(
        pipeline_result: Any,
    ) -> list[Any]:
        """
        Normalize the FacePipeline output into a list.

        Expected normal case:

            List[FaceRecord]
        """

        if pipeline_result is None:
            return []

        if isinstance(
            pipeline_result,
            (list, tuple),
        ):

            return list(
                pipeline_result
            )

        if isinstance(
            pipeline_result,
            dict,
        ):

            for key in (
                "faces",
                "face_records",
                "results",
            ):

                if key in pipeline_result:

                    value = (
                        pipeline_result[key]
                    )

                    if value is None:
                        return []

                    return list(
                        value
                    )

            raise TypeError(
                "FacePipeline returned a dictionary "
                "without faces/face_records/results."
            )

        if hasattr(
            pipeline_result,
            "faces",
        ):

            value = (
                pipeline_result.faces
            )

            if value is None:
                return []

            return list(
                value
            )

        raise TypeError(
            "Unsupported FacePipeline result type: "
            f"{type(pipeline_result)!r}"
        )

    # ============================================================
    # FACE PERSISTENCE
    # ============================================================

    @staticmethod
    def _persist_face(
        *,
        face_repo: FaceRepository,
        image_id: int,
        face_index: int,
        face_data: Any,
    ) -> Any:
        """
        Convert one FaceRecord into the PostgreSQL faces row.
        """

        detection_confidence = (
            ProcessingService._get_value(
                face_data,
                "detection_confidence",
                default=None,
            )
        )

        if detection_confidence is None:

            raise ValueError(
                f"Face {face_index} has no "
                "detection_confidence."
            )

        bbox = (
            ProcessingService._get_value(
                face_data,
                "bbox",
                default=None,
            )
        )

        if bbox is None:

            raise ValueError(
                f"Face {face_index} has no bbox."
            )

        bbox = list(
            np.asarray(
                bbox,
                dtype=np.float32,
            ).reshape(-1)
        )

        if len(bbox) != 4:

            raise ValueError(
                f"Face {face_index} bbox must contain "
                f"4 values; received {len(bbox)}."
            )

        landmarks = (
            ProcessingService._get_value(
                face_data,
                "landmarks",
                default=None,
            )
        )

        if landmarks is not None:

            landmarks = (
                ProcessingService._flatten_landmarks(
                    landmarks
                )
            )

        gender_result = (
            ProcessingService._get_value(
                face_data,
                "gender",
                default=None,
            )
        )

        age_result = (
            ProcessingService._get_value(
                face_data,
                "age",
                default=None,
            )
        )

        gender = (
            ProcessingService._extract_nested_value(
                gender_result,
                "label",
                "value",
                "gender",
            )
        )

        gender_confidence = (
            ProcessingService._extract_nested_value(
                gender_result,
                "confidence",
                "score",
            )
        )

        age_years = (
            ProcessingService._extract_nested_value(
                age_result,
                "years",
                "age_years",
                "value",
            )
        )

        age_group = (
            ProcessingService._extract_nested_value(
                age_result,
                "group",
                "age_group",
                "label",
            )
        )

        age_confidence = (
            ProcessingService._extract_nested_value(
                age_result,
                "confidence",
                "score",
            )
        )

        # --------------------------------------------------------
        # Some FaceRecord implementations may expose flattened
        # fields directly. Prefer those if the nested result
        # did not provide a value.
        # --------------------------------------------------------

        if gender is None:

            gender = (
                ProcessingService._get_value(
                    face_data,
                    "gender_label",
                    default=None,
                )
            )

        if gender_confidence is None:

            gender_confidence = (
                ProcessingService._get_value(
                    face_data,
                    "gender_confidence",
                    default=None,
                )
            )

        if age_years is None:

            age_years = (
                ProcessingService._get_value(
                    face_data,
                    "age_years",
                    default=None,
                )
            )

        if age_group is None:

            age_group = (
                ProcessingService._get_value(
                    face_data,
                    "age_group",
                    default=None,
                )
            )

        if age_confidence is None:

            age_confidence = (
                ProcessingService._get_value(
                    face_data,
                    "age_confidence",
                    default=None,
                )
            )

        return face_repo.create_or_update(
            image_id=image_id,
            face_index=face_index,
            detection_confidence=float(
                detection_confidence
            ),
            bbox=bbox,
            landmarks=landmarks,
            gender=(
                str(gender)
                if gender is not None
                else None
            ),
            gender_confidence=(
                float(gender_confidence)
                if gender_confidence is not None
                else None
            ),
            age_years=(
                int(age_years)
                if age_years is not None
                else None
            ),
            age_group=(
                str(age_group)
                if age_group is not None
                else None
            ),
            age_confidence=(
                float(age_confidence)
                if age_confidence is not None
                else None
            ),
        )

    # ============================================================
    # EMBEDDING
    # ============================================================

    @staticmethod
    def _extract_embedding(
        face_data: Any,
    ) -> np.ndarray:
        """
        Extract and validate the 512-D ArcFace embedding.
        """

        embedding = (
            ProcessingService._get_value(
                face_data,
                "embedding",
                default=None,
            )
        )

        if embedding is None:

            raise ValueError(
                "FaceRecord does not contain an embedding."
            )

        vector = np.asarray(
            embedding,
            dtype=np.float32,
        ).reshape(-1)

        if vector.size != EMBEDDING_DIMENSION:

            raise ValueError(
                "ArcFace embedding dimension mismatch: "
                f"received={vector.size}, "
                f"expected={EMBEDDING_DIMENSION}"
            )

        if not np.isfinite(
            vector
        ).all():

            raise ValueError(
                "Embedding contains NaN or infinite values."
            )

        norm = float(
            np.linalg.norm(
                vector
            )
        )

        if norm <= 0.0:

            raise ValueError(
                "Embedding has zero norm."
            )

        return vector

    # ============================================================
    # REDIS - UPLOAD STATE
    # ============================================================

    def _set_upload_state(
        self,
        *,
        event: UploadEvent,
        status: str,
        upload_id: int | None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """
        Redis is operational/transient state.

        PostgreSQL remains the authoritative state database.
        """

        key = (
            f"workflow1:"
            f"upload:{event.event_id}"
        )

        payload: dict[str, Any] = {
            "event_id": str(
                event.event_id
            ),
            "workflow": WORKFLOW_NAME,
            "status": status,
        }

        if upload_id is not None:

            payload[
                "upload_id"
            ] = upload_id

        if extra:

            payload.update(
                extra
            )

        try:

            self.redis_service.set(
                key,
                payload,
                ttl=REDIS_STATE_TTL,
            )

        except Exception:

            logger.exception(
                "Redis upload-state update failed. "
                "Continuing because Redis is auxiliary."
            )

    # ============================================================
    # REDIS - FACE STATE
    # ============================================================

    def _set_face_state(
        self,
        *,
        face: Any,
        image_id: int,
        upload_id: int,
        gender: Any,
        age_group: Any,
    ) -> None:
        """
        Store lightweight live face metadata in Redis.

        PostgreSQL remains authoritative.
        """

        try:

            self.redis_service.hset(
                "workflow1:faces",
                f"face:{face.id}",
                {
                    "face_id": face.id,
                    "image_id": image_id,
                    "upload_id": upload_id,
                    "gender": (
                        str(gender)
                        if gender is not None
                        else None
                    ),
                    "age_group": (
                        str(age_group)
                        if age_group is not None
                        else None
                    ),
                },
            )

        except Exception:

            logger.exception(
                "Redis face-state update failed. "
                "Continuing because Redis is auxiliary."
            )

    # ============================================================
    # VALUE HELPERS
    # ============================================================

    @staticmethod
    def _get_value(
        obj: Any,
        name: str,
        default: Any = None,
    ) -> Any:
        """
        Read one field from either a Pydantic/model object
        or a dictionary.
        """

        if obj is None:

            return default

        if isinstance(
            obj,
            dict,
        ):

            return obj.get(
                name,
                default,
            )

        return getattr(
            obj,
            name,
            default,
        )

    @staticmethod
    def _extract_nested_value(
        obj: Any,
        *names: str,
    ) -> Any:
        """
        Extract a value from nested GenderResult/AgeResult
        style objects or dictionaries.
        """

        if obj is None:

            return None

        if isinstance(
            obj,
            dict,
        ):

            for name in names:

                if name in obj:

                    return obj[name]

            return None

        for name in names:

            if hasattr(
                obj,
                name,
            ):

                return getattr(
                    obj,
                    name,
                )

        # --------------------------------------------------------
        # Pydantic models may expose model_dump().
        # --------------------------------------------------------

        if hasattr(
            obj,
            "model_dump",
        ):

            try:

                dumped = obj.model_dump()

                for name in names:

                    if name in dumped:

                        return dumped[name]

            except Exception:

                pass

        return None

    @staticmethod
    def _flatten_landmarks(
        landmarks: Any,
    ) -> list[float]:
        """
        Convert five landmarks into PostgreSQL's ten coordinate
        columns.

        Expected:

            [[x1,y1], [x2,y2], ...]

        or:

            [x1,y1,x2,y2,...]
        """

        array = np.asarray(
            landmarks,
            dtype=np.float32,
        )

        if array.size != 10:

            raise ValueError(
                "Expected 5 facial landmarks "
                "(10 coordinate values), "
                f"received {array.size}."
            )

        return [
            float(value)
            for value in array.reshape(-1)
        ]

    @staticmethod
    def _extract_gender(
        face_data: Any,
    ) -> Any:

        gender_result = (
            ProcessingService._get_value(
                face_data,
                "gender",
                default=None,
            )
        )

        value = (
            ProcessingService._extract_nested_value(
                gender_result,
                "label",
                "value",
                "gender",
            )
        )

        if value is not None:
            return value

        return (
            ProcessingService._get_value(
                face_data,
                "gender_label",
                default=gender_result,
            )
        )

    @staticmethod
    def _extract_age_group(
        face_data: Any,
    ) -> Any:

        age_result = (
            ProcessingService._get_value(
                face_data,
                "age",
                default=None,
            )
        )

        value = (
            ProcessingService._extract_nested_value(
                age_result,
                "group",
                "age_group",
                "label",
            )
        )

        if value is not None:
            return value

        return (
            ProcessingService._get_value(
                face_data,
                "age_group",
                default=None,
            )
        )