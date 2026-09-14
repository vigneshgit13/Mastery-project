from __future__ import annotations

import logging

from google.cloud import pubsub_v1

from app.core.config import PROJECT_ID
from app.db.postgres import SessionLocal
from app.db.repository import Workflow1OutboxRepository


logger = logging.getLogger(__name__)


WORKFLOW1_COMPLETION_TOPIC = "wf1-batch-completed"


class Workflow1OutboxPublisher:
    """
    Publishes committed Workflow 1 outbox events to Pub/Sub.

    Responsibilities
    ----------------
    1. Read PENDING events from PostgreSQL.
    2. Publish the exact persisted payload to Pub/Sub.
    3. Mark the event PUBLISHED only after Pub/Sub confirms success.
    4. Mark the event FAILED if publishing fails.

    It does NOT:
        - perform face processing
        - modify Workflow 1 processing state
        - create outbox events
        - ACK/NACK incoming Pub/Sub messages
    """

    def __init__(
        self,
        *,
        project_id: str = PROJECT_ID,
        topic_name: str = WORKFLOW1_COMPLETION_TOPIC,
    ) -> None:

        self.publisher = pubsub_v1.PublisherClient()

        self.topic_path = (
            self.publisher.topic_path(
                project_id,
                topic_name,
            )
        )

        logger.info(
            "Workflow 1 Outbox Publisher initialized: "
            "topic=%s",
            self.topic_path,
        )

    def publish_pending(
        self,
        *,
        limit: int = 100,
    ) -> int:
        """
        Publish pending Workflow 1 outbox events.

        Returns
        -------
        int
            Number of successfully published events.
        """

        db = SessionLocal()

        published_count = 0

        try:

            repository = Workflow1OutboxRepository(db)

            events = repository.get_pending(
                limit=limit
            )

            logger.info(
                "Found %d pending Workflow 1 outbox event(s).",
                len(events),
            )

            for event in events:

                logger.info(
                    "Publishing outbox event: "
                    "id=%s event_id=%s event_type=%s",
                    event.id,
                    event.event_id,
                    event.event_type,
                )

                try:

                    # --------------------------------------------------
                    # Record the attempt.
                    # --------------------------------------------------

                    repository.mark_attempt(
                        event
                    )

                    db.commit()

                    # --------------------------------------------------
                    # Publish the exact persisted payload.
                    # --------------------------------------------------

                    future = self.publisher.publish(
                        self.topic_path,
                        event.payload.encode(
                            "utf-8"
                        ),
                    )

                    message_id = future.result()

                    logger.info(
                        "Pub/Sub publish successful: "
                        "event_id=%s message_id=%s",
                        event.event_id,
                        message_id,
                    )

                    # --------------------------------------------------
                    # IMPORTANT:
                    #
                    # Only mark PUBLISHED after Pub/Sub confirms
                    # successful publication.
                    # --------------------------------------------------

                    repository.mark_published(
                        event
                    )

                    db.commit()

                    published_count += 1

                    logger.info(
                        "Outbox event marked PUBLISHED: "
                        "event_id=%s",
                        event.event_id,
                    )

                except Exception as exc:

                    logger.exception(
                        "Failed to publish outbox event: "
                        "event_id=%s",
                        event.event_id,
                    )

                    db.rollback()

                    try:

                        repository = (
                            Workflow1OutboxRepository(
                                db
                            )
                        )

                        # Re-load the event because the previous
                        # transaction was rolled back.
                        pending_events = (
                            repository.get_pending(
                                limit=limit
                            )
                        )

                        failed_event = next(
                            (
                                item
                                for item in pending_events
                                if item.id == event.id
                            ),
                            None,
                        )

                        if failed_event is not None:

                            repository.mark_failed(
                                failed_event,
                                str(exc),
                            )

                            db.commit()

                    except Exception:

                        logger.exception(
                            "Unable to persist outbox failure state: "
                            "event_id=%s",
                            event.event_id,
                        )

                        db.rollback()

            return published_count

        finally:

            db.close()

    def close(self) -> None:
        """Close the Pub/Sub publisher client."""

        self.publisher.transport.close()