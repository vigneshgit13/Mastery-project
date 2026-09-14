import logging
import time

from app.publisher.workflow1_outbox_publisher import (
    Workflow1OutboxPublisher,
)

logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=" * 70)
    logger.info("WORKFLOW 1 OUTBOX PUBLISHER")
    logger.info("=" * 70)

    publisher = Workflow1OutboxPublisher()

    try:
        while True:
            try:
                published = publisher.publish_pending(limit=100)

                if published > 0:
                    logger.info(
                        "Published %d Workflow 1 outbox event(s).",
                        published,
                    )

            except Exception:
                logger.exception(
                    "Workflow 1 outbox publishing cycle failed."
                )

            time.sleep(1)

    except KeyboardInterrupt:
        logger.info(
            "Workflow 1 outbox publisher stopped."
        )

    finally:
        publisher.close()


if __name__ == "__main__":
    main()