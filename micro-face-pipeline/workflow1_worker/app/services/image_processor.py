import logging
from pathlib import Path

from app.ai.pipeline_factory import create_face_pipeline
from app.models.face_record import FaceRecord


logger = logging.getLogger(__name__)


class ImageProcessor:

    def __init__(self):

        logger.info(
            "Initializing ImageProcessor..."
        )

        self.pipeline = (
            create_face_pipeline()
        )

        logger.info(
            "ImageProcessor ready."
        )

    def process(
        self,
        image_path: str | Path,
    ) -> list[FaceRecord]:

        logger.info("=" * 80)
        logger.info(
            "PROCESSING DOWNLOADED IMAGE"
        )
        logger.info(
            "Image: %s",
            image_path,
        )

        records = self.pipeline.process(
            image_path
        )

        logger.info(
            "AI processing complete. "
            "Faces extracted: %d",
            len(records),
        )

        return records