from __future__ import annotations

import logging

from app.ai.pipeline_factory import create_face_pipeline
from app.services.processing_service import ProcessingService
from app.subscriber.pubsub_subscriber import PubSubSubscriber


logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)


def main() -> None:

    logger = logging.getLogger(__name__)

    logger.info(
        "Initializing Workflow 1..."
    )

    # ------------------------------------------------------------
    # Existing AI pipeline
    # ------------------------------------------------------------

    pipeline = create_face_pipeline()

    # ------------------------------------------------------------
    # Workflow 1 orchestration
    # ------------------------------------------------------------

    processing_service = ProcessingService(
        pipeline=pipeline,
    )

    # ------------------------------------------------------------
    # Pub/Sub adapter
    # ------------------------------------------------------------

    subscriber = PubSubSubscriber(
        processing_service=processing_service,
    )

    logger.info(
        "Workflow 1 initialized successfully."
    )

    subscriber.start()


if __name__ == "__main__":
    main()