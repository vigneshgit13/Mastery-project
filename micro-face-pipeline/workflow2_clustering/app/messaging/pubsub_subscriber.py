from __future__ import annotations

import json
import logging
from typing import Any

from app.clustering.clustering_processor import (
    ClusteringProcessor,
)
from app.models.face_extraction_completed import (
    FaceExtractionCompletedEvent,
)


logger = logging.getLogger(__name__)


class Workflow2PubSubSubscriber:
    """
    Thin Workflow 2 Pub/Sub transport adapter.

    Responsibilities
    ----------------
    1. Decode Pub/Sub message bytes.
    2. Parse the JSON payload.
    3. Construct the domain event.
    4. Validate the domain event.
    5. Delegate business processing to ClusteringProcessor.
    6. ACK only after successful processing.
    7. NACK when processing fails.

    This class does NOT:
        - query PostgreSQL directly
        - manipulate HNSW
        - perform clustering
        - create clusters
        - create memberships
        - implement similarity logic
    """

    def __init__(
        self,
        processor: ClusteringProcessor,
    ) -> None:
        self.processor = processor

    # ==================================================================
    # PUB/SUB CALLBACK
    # ==================================================================

    def callback(
        self,
        message: Any,
    ) -> None:
        """
        Google Cloud Pub/Sub callback.

        Expected message interface:
            message.data
            message.ack()
            message.nack()
        """

        try:
            event = self._decode_event(
                message.data
            )

            result = self.processor.process(
                event
            )

            if not result.success:
                raise RuntimeError(
                    "ClusteringProcessor reported unsuccessful "
                    f"processing: status={result.status}"
                )

            message.ack()

            logger.info(
                "Workflow 2 Pub/Sub message ACKed successfully: "
                "event_id=%s upload_id=%s image_id=%s ",
                event.event_id,
                event.upload_id,
                event.image_id,
            )

        except Exception:
            logger.exception(
                "Workflow 2 Pub/Sub message processing failed; "
                "message will be NACKed."
            )

            message.nack()

    # ==================================================================
    # EVENT DECODING
    # ==================================================================

    @staticmethod
    def _decode_event(
        data: bytes,
    ) -> FaceExtractionCompletedEvent:
        """
        Convert Pub/Sub message bytes into the validated
        Workflow 2 domain event.
        """

        if not data:
            raise ValueError(
                "Pub/Sub message contains empty data."
            )

        # --------------------------------------------------------------
        # UTF-8 decoding
        # --------------------------------------------------------------

        try:
            payload_text = data.decode(
                "utf-8"
            )

        except UnicodeDecodeError as exc:
            raise ValueError(
                "Pub/Sub message is not valid UTF-8."
            ) from exc

        # --------------------------------------------------------------
        # JSON decoding
        # --------------------------------------------------------------

        try:
            payload = json.loads(
                payload_text
            )

        except json.JSONDecodeError as exc:
            raise ValueError(
                "Pub/Sub message does not contain valid JSON."
            ) from exc

        # --------------------------------------------------------------
        # Payload type
        # --------------------------------------------------------------

        if not isinstance(
            payload,
            dict,
        ):
            raise ValueError(
                "Pub/Sub event payload must be a JSON object."
            )

        # --------------------------------------------------------------
        # Domain event validation
        #
        # Pydantic enforces:
        #   - required fields
        #   - field types
        #   - UUID format
        #   - face_count >= 0
        #   - extra='forbid'
        # --------------------------------------------------------------

        event = (
            FaceExtractionCompletedEvent
            .model_validate(
                payload
            )
        )

        # --------------------------------------------------------------
        # Semantic event-type validation
        # --------------------------------------------------------------

        event.validate_event_type()

        return event
