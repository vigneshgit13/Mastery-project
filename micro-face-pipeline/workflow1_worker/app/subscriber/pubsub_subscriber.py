from __future__ import annotations

import json
import logging

from google.cloud import pubsub_v1

from app.ai.pipeline_factory import create_face_pipeline
from app.core.config import (
    PROJECT_ID,
    PUBSUB_SUBSCRIPTION,
)
from app.models.upload_event import UploadEvent
from app.services.processing_service import ProcessingService


logger = logging.getLogger(__name__)


class PubSubSubscriber:
    """
    Thin Pub/Sub adapter.

    Responsibilities:

        Pub/Sub
            ↓
        decode
            ↓
        validate UploadEvent
            ↓
        ProcessingService
            ↓
        ACK / NACK
    """

    def __init__(
        self,
        processing_service: ProcessingService,
    ) -> None:

        self.processing_service = (
            processing_service
        )

        self.subscriber = (
            pubsub_v1.SubscriberClient()
        )

        self.subscription_path = (
            self.subscriber.subscription_path(
                PROJECT_ID,
                PUBSUB_SUBSCRIPTION,
            )
        )

        logger.info(
            "Pub/Sub subscriber initialized."
        )

        logger.info(
            "Subscription: %s",
            self.subscription_path,
        )

    # ==================================================================
    # CALLBACK
    # ==================================================================

    def callback(
        self,
        message,
    ) -> None:

        logger.info("=" * 80)
        logger.info("PUB/SUB MESSAGE RECEIVED")
        logger.info(
            "Message ID: %s",
            message.message_id,
        )
        logger.info("=" * 80)

        try:

            # ----------------------------------------------------------
            # Decode
            # ----------------------------------------------------------

            payload = json.loads(
                message.data.decode(
                    "utf-8"
                )
            )

            # ----------------------------------------------------------
            # Validate
            # ----------------------------------------------------------

            event = (
                UploadEvent.model_validate(
                    payload
                )
            )

            logger.info(
                "UploadEvent validated."
            )

            logger.info(
                "Event ID : %s",
                event.event_id,
            )

            logger.info(
                "Bucket   : %s",
                event.bucket,
            )

            logger.info(
                "Blob     : %s",
                event.blob_name,
            )

            # ----------------------------------------------------------
            # Business processing
            # ----------------------------------------------------------

            result = (
                self.processing_service.process_upload(
                    event
                )
            )

            logger.info(
                "Processing result: %s",
                result,
            )

            # ----------------------------------------------------------
            # ACK ONLY AFTER COMPLETE SUCCESS
            # ----------------------------------------------------------

            message.ack()

            logger.info(
                "Message ACKed: %s",
                message.message_id,
            )

        except json.JSONDecodeError:

            logger.error(
                "Invalid JSON received. "
                "ACKing because retry cannot fix malformed JSON."
            )

            message.ack()

        except Exception:

            logger.exception(
                "Workflow 1 processing failed. "
                "Message will be retried."
            )

            message.nack()

    # ==================================================================
    # START
    # ==================================================================

    def start(
        self,
    ) -> None:

        logger.info("=" * 80)
        logger.info(
            "STARTING WORKFLOW 1 PUB/SUB SUBSCRIBER"
        )
        logger.info(
            "Subscription: %s",
            self.subscription_path,
        )
        logger.info("=" * 80)

        streaming_future = (
            self.subscriber.subscribe(
                self.subscription_path,
                callback=self.callback,
            )
        )

        try:

            streaming_future.result()

        except KeyboardInterrupt:

            logger.info(
                "Stopping subscriber..."
            )

            streaming_future.cancel()

        except Exception:

            logger.exception(
                "Subscriber stopped unexpectedly."
            )

            streaming_future.cancel()

            raise

        finally:

            self.subscriber.close()


def create_subscriber() -> PubSubSubscriber:

    logger.info("=" * 80)
    logger.info(
        "INITIALIZING WORKFLOW 1"
    )
    logger.info("=" * 80)

    # --------------------------------------------------------------
    # AI pipeline
    # --------------------------------------------------------------

    pipeline = (
        create_face_pipeline()
    )

    # --------------------------------------------------------------
    # Processing service
    # --------------------------------------------------------------

    processing_service = (
        ProcessingService(
            pipeline=pipeline
        )
    )

    # --------------------------------------------------------------
    # Subscriber
    # --------------------------------------------------------------

    return PubSubSubscriber(
        processing_service=processing_service
    )


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
    )

    subscriber = (
        create_subscriber()
    )

    subscriber.start()