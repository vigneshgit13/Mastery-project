from __future__ import annotations

from dataclasses import dataclass
import logging

from app.clustering.clustering_service import (
    ClusterAssignment,
    ClusteringService,
)
from app.db.clustering_repository import (
    ClusteringRepository,
)
from app.models.face_extraction_completed import (
    FaceExtractionCompletedEvent,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClusteringProcessResult:
    """
    Result of processing one Workflow 1 completion event.
    """

    status: str

    event_id: str

    upload_id: int

    image_id: int

    expected_face_count: int

    processed_face_count: int

    created_cluster_count: int

    existing_cluster_count: int

    assignments: tuple[ClusterAssignment, ...]

    @property
    def success(self) -> bool:
        return self.status == "COMPLETED"


class ClusteringProcessor:
    """
    Workflow 2 business/orchestration layer.

    Responsibilities
    ----------------
    1. Validate the Workflow 1 completion event.
    2. Read authoritative embedding metadata from PostgreSQL.
    3. Restrict processing to the event's image_id.
    4. Pass each embedding record to ClusteringService.
    5. Validate the resulting face count.
    6. Return a deterministic processing result.

    It does NOT:
        - consume Pub/Sub messages
        - ACK/NACK messages
        - directly manipulate FAISS/HNSW
        - implement similarity logic
        - create clusters directly

    Those responsibilities belong to their respective layers.
    """

    def __init__(
        self,
        repository: ClusteringRepository,
        clustering_service: ClusteringService,
    ) -> None:

        self.repository = repository
        self.clustering_service = clustering_service

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    def process(
        self,
        event: FaceExtractionCompletedEvent,
    ) -> ClusteringProcessResult:

        event.validate_event_type()

        # --------------------------------------------------------------
        # Explicit event-level idempotency.
        # --------------------------------------------------------------

        if self.repository.is_event_completed(
            str(event.event_id)
        ):
            logger.info(
                "Workflow 2 event already completed; "
                "skipping duplicate event: event_id=%s "
                "upload_id=%s image_id=%s",
                event.event_id,
                event.upload_id,
                event.image_id,
            )

            return ClusteringProcessResult(
                status="COMPLETED",
                event_id=str(event.event_id),
                upload_id=event.upload_id,
                image_id=event.image_id,
                expected_face_count=event.face_count,
                processed_face_count=event.face_count,
                created_cluster_count=0,
                existing_cluster_count=event.face_count,
                assignments=tuple(),
            )

        logger.info(
            "Starting Workflow 2 clustering: "
            "event_id=%s upload_id=%s image_id=%s "
            "expected_faces=%s",
            event.event_id,
            event.upload_id,
            event.image_id,
            event.face_count,
        )


        logger.info(
            "Starting Workflow 2 clustering: "
            "event_id=%s upload_id=%s image_id=%s "
            "expected_faces=%s",
            event.event_id,
            event.upload_id,
            event.image_id,
            event.face_count,
        )

        records = (
            self.repository
            .get_face_embeddings_for_image(
                event.image_id
            )
        )

        actual_face_count = len(records)

        # --------------------------------------------------------------
        # Producer/consumer consistency check.
        # --------------------------------------------------------------

        if actual_face_count != event.face_count:

            raise RuntimeError(
                "Workflow 1 completion event does not match "
                "PostgreSQL embedding state: "
                f"image_id={event.image_id}, "
                f"event_face_count={event.face_count}, "
                f"database_face_count={actual_face_count}"
            )

        assignments: list[ClusterAssignment] = []

        for record in records:

            assignment = (
                self.clustering_service
                .process_face(
                    record
                )
            )

            assignments.append(
                assignment
            )

        created_count = sum(
            1
            for assignment in assignments
            if assignment.created_new_cluster
        )

        existing_count = (
            len(assignments)
            - created_count
        )

        # --------------------------------------------------------------
        # Mark the event as COMPLETED only after every face has been
        # processed successfully.
        # --------------------------------------------------------------

        self.repository.mark_event_completed(
            event_id=str(event.event_id),
            event_type=str(event.event_type),
            workflow="workflow2",
            upload_id=event.upload_id,
            image_id=event.image_id,
        )

        logger.info(
            "Workflow 2 clustering completed: "
            "event_id=%s image_id=%s faces=%s "
            "new_clusters=%s existing_clusters=%s",
            event.event_id,
            event.image_id,
            actual_face_count,
            created_count,
            existing_count,
        )

        return ClusteringProcessResult(
            status="COMPLETED",
            event_id=str(event.event_id),
            upload_id=event.upload_id,
            image_id=event.image_id,
            expected_face_count=event.face_count,
            processed_face_count=actual_face_count,
            created_cluster_count=created_count,
            existing_cluster_count=existing_count,
            assignments=tuple(assignments),
        )