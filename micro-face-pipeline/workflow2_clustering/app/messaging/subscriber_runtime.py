from __future__ import annotations

import logging

from google.cloud import pubsub_v1

from app.clustering.clustering_processor import ClusteringProcessor
from app.core.config import PROJECT_ID, PUBSUB_SUBSCRIPTION
from app.messaging.pubsub_subscriber import Workflow2PubSubSubscriber


logger = logging.getLogger(__name__)


class Workflow2SubscriberRuntime:
    """
    Google Cloud Pub/Sub runtime for Workflow 2.

    Responsibilities:
        - create SubscriberClient
        - construct subscription path
        - connect the subscription to the thin subscriber callback
        - keep the pull subscriber running
        - support graceful shutdown

    Business logic remains inside ClusteringProcessor.
    """

    def __init__(
        self,
        processor: ClusteringProcessor,
    ) -> None:

        self.processor = processor

        self.message_subscriber = Workflow2PubSubSubscriber(
            processor=processor
        )

        self.subscriber = pubsub_v1.SubscriberClient()

        self.subscription_path = self.subscriber.subscription_path(
            PROJECT_ID,
            PUBSUB_SUBSCRIPTION,
        )

    def run(self) -> None:
        """
        Start the blocking Pub/Sub pull loop.
        """

        logger.info(
            "Starting Workflow 2 Pub/Sub subscriber."
        )

        logger.info(
            "Project: %s",
            PROJECT_ID,
        )

        logger.info(
            "Subscription: %s",
            self.subscription_path,
        )

        streaming_pull_future = self.subscriber.subscribe(
            self.subscription_path,
            callback=self.message_subscriber.callback,
        )

        logger.info(
            "Workflow 2 Pub/Sub subscriber is listening."
        )

        try:
            streaming_pull_future.result()

        except KeyboardInterrupt:

            logger.info(
                "KeyboardInterrupt received. "
                "Stopping Workflow 2 Pub/Sub subscriber..."
            )

            streaming_pull_future.cancel()

            try:
                streaming_pull_future.result(timeout=10)

            except Exception:
                pass

        except Exception:

            logger.exception(
                "Workflow 2 Pub/Sub subscriber stopped unexpectedly."
            )

            raise

        finally:

            logger.info(
                "Closing Workflow 2 Pub/Sub subscriber client..."
            )

            self.subscriber.close()

            logger.info(
                "Workflow 2 Pub/Sub subscriber stopped."
            )