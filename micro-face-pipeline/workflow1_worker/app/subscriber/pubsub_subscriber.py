import json
import logging

from google.cloud import pubsub_v1

from app.core.config import (
    PROJECT_ID,
    PUBSUB_SUBSCRIPTION,
)

from app.models.upload_event import UploadEvent

from app.services.gcs_service import (
    gcs_service,
)

from app.ai.detector import (
    SCRFDDetector,
)

from app.ai.aligner import (
    FaceAligner,
)

from app.ai.recognizer import (
    ArcFaceRecognizer,
)

from app.ai.attributes import (
    GenderAgeClassifier,
)

from app.ai.pipeline import (
    FacePipeline,
)


logging.basicConfig(
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


class PubSubSubscriber:

    def __init__(self):

        # ==========================================================
        # PUB/SUB
        # ==========================================================

        self.subscriber = (
            pubsub_v1.SubscriberClient()
        )

        self.subscription_path = (
            self.subscriber.subscription_path(
                PROJECT_ID,
                PUBSUB_SUBSCRIPTION,
            )
        )

        # ==========================================================
        # LOAD AI MODELS ONCE
        # ==========================================================

        logger.info("=" * 80)
        logger.info("INITIALIZING AI MODELS")
        logger.info("=" * 80)

        self.detector = SCRFDDetector(
            model_path=(
                "models/scrfd_10g_bnkps.onnx"
            ),
            input_size=640,
            confidence_threshold=0.5,
            nms_threshold=0.4,
        )

        self.aligner = FaceAligner()

        self.recognizer = ArcFaceRecognizer(
            model_path=(
                "models/glintr100.onnx"
            ),
        )

        self.classifier = GenderAgeClassifier(
            model_path=(
                "models/model.onnx"
            ),
        )

        # ==========================================================
        # FACE PIPELINE
        # ==========================================================

        self.pipeline = FacePipeline(
            detector=self.detector,
            aligner=self.aligner,
            recognizer=self.recognizer,
            classifier=self.classifier,
        )

        logger.info(
            "AI models initialized successfully"
        )

        logger.info("=" * 80)

    # ==================================================================
    # PUB/SUB CALLBACK
    # ==================================================================

    def callback(
        self,
        message,
    ):

        try:

            # ======================================================
            # 1. READ MESSAGE
            # ======================================================

            text = message.data.decode(
                "utf-8"
            )

            logger.info("=" * 80)
            logger.info(
                "RECEIVED PUB/SUB MESSAGE"
            )
            logger.info("=" * 80)

            logger.info(
                "%s",
                text,
            )

            # ======================================================
            # 2. PARSE JSON
            # ======================================================

            data = json.loads(
                text
            )

            # ======================================================
            # 3. VALIDATE UPLOAD EVENT
            # ======================================================

            upload_event = (
                UploadEvent.model_validate(
                    data
                )
            )

            logger.info(
                "UploadEvent validated"
            )

            logger.info(
                "Event ID    : %s",
                upload_event.event_id,
            )

            logger.info(
                "Bucket      : %s",
                upload_event.bucket,
            )

            logger.info(
                "Blob        : %s",
                upload_event.blob_name,
            )

            # ======================================================
            # 4. DOWNLOAD IMAGE FROM GCS
            # ======================================================

            local_path = (
                gcs_service.download_file(
                    bucket_name=(
                        upload_event.bucket
                    ),
                    blob_name=(
                        upload_event.blob_name
                    ),
                )
            )

            logger.info(
                "Downloaded image: %s",
                local_path,
            )

            # ======================================================
            # 5. RUN COMPLETE AI PIPELINE
            # ======================================================

            logger.info("=" * 80)
            logger.info(
                "RUNNING FACE PIPELINE"
            )
            logger.info("=" * 80)

            records = (
                self.pipeline.process(
                    local_path
                )
            )

            # ======================================================
            # 6. PIPELINE RESULT
            # ======================================================

            logger.info("=" * 80)
            logger.info(
                "PIPELINE RESULT"
            )
            logger.info("=" * 80)

            logger.info(
                "Total faces: %d",
                len(records),
            )

            for record in records:

                logger.info(
                    "Face %02d | "
                    "Gender: %s (%.4f) | "
                    "Age: %d | "
                    "Group: %s | "
                    "Embedding: %d-D",

                    record.face_index,

                    record.gender.label,

                    record.gender.confidence,

                    record.age.years,

                    record.age.group,

                    len(record.embedding),
                )

            # ======================================================
            # 7. ACK ONLY AFTER COMPLETE SUCCESS
            # ======================================================

            message.ack()

            logger.info(
                "Message ACKed"
            )

            logger.info("=" * 80)

        # ==========================================================
        # INVALID JSON
        # ==========================================================

        except json.JSONDecodeError:

            logger.error(
                "Invalid JSON received."
            )

            # Invalid message cannot be fixed
            # by retrying.

            message.ack()

        # ==========================================================
        # ANY PROCESSING FAILURE
        # ==========================================================

        except Exception:

            logger.exception(
                "Processing failed. "
                "Message will be retried."
            )

            message.nack()

    # ==================================================================
    # START SUBSCRIBER
    # ==================================================================

    def start(
        self,
    ):

        logger.info("=" * 80)
        logger.info(
            "STARTING PUB/SUB SUBSCRIBER"
        )
        logger.info(
            "Subscription: %s",
            self.subscription_path,
        )
        logger.info("=" * 80)

        future = (
            self.subscriber.subscribe(
                self.subscription_path,
                callback=self.callback,
            )
        )

        future.result()


subscriber = PubSubSubscriber()