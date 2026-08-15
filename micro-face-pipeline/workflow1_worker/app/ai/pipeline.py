import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np

from app.ai.detector import SCRFDDetector
from app.ai.aligner import FaceAligner
from app.ai.recognizer import ArcFaceRecognizer
from app.ai.attributes import GenderAgeClassifier

from app.models.face_record import (
    FaceRecord,
    GenderResult,
    AgeResult,
)


logger = logging.getLogger(__name__)


class FacePipeline:
    """
    Complete Workflow 1 face-processing pipeline.

    Processing flow:

        Image
          ↓
        SCRFD detection
          ↓
        Face alignment
          ↓
        ArcFace 512-D embedding
          ↓
        Age + Gender classification
          ↓
        FaceRecord[]
    """

    def __init__(
        self,
        detector: SCRFDDetector,
        aligner: FaceAligner,
        recognizer: ArcFaceRecognizer,
        classifier: GenderAgeClassifier,
    ):
        self.detector = detector
        self.aligner = aligner
        self.recognizer = recognizer
        self.classifier = classifier

        logger.info("=" * 80)
        logger.info("FACE PIPELINE INITIALIZED")
        logger.info("=" * 80)

    # ==================================================================
    # PROCESS IMAGE
    # ==================================================================

    def process(
        self,
        image_path: str | Path,
    ) -> List[FaceRecord]:

        image_path = Path(image_path)

        logger.info("=" * 80)
        logger.info("STARTING FACE PIPELINE")
        logger.info("Image: %s", image_path)
        logger.info("=" * 80)

        # ==================================================================
        # STEP 0 — LOAD IMAGE
        # ==================================================================

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            raise RuntimeError(
                f"Unable to read image: {image_path}"
            )

        logger.info(
            "Image loaded successfully"
        )

        logger.info(
            "Image shape: %s",
            image.shape,
        )

        # ==================================================================
        # STEP 1 — FACE DETECTION
        # ==================================================================

        logger.info("-" * 80)
        logger.info("STEP 1: FACE DETECTION")
        logger.info("-" * 80)

        detections = self.detector.detect(
            image
        )

        logger.info(
            "Faces detected: %d",
            len(detections),
        )

        if not detections:

            logger.warning(
                "No faces detected in image: %s",
                image_path,
            )

            return []

        # ==================================================================
        # STEP 2 — PROCESS EACH FACE
        # ==================================================================

        records: List[FaceRecord] = []

        for index, detection in enumerate(
            detections,
            start=1,
        ):

            logger.info("=" * 80)
            logger.info(
                "PROCESSING FACE %d/%d",
                index,
                len(detections),
            )
            logger.info("=" * 80)

            # ==============================================================
            # DETECTION INFORMATION
            # ==============================================================

            logger.info(
                "Detection confidence: %.4f",
                detection.score,
            )

            logger.info(
                "Bounding box: %s",
                detection.bbox,
            )

            # ==============================================================
            # STEP 2A — FACE ALIGNMENT
            # ==============================================================

            logger.info(
                "Aligning face..."
            )

            try:

                aligned_face = self.aligner.align(
                    image,
                    detection,
                )

            except Exception:

                logger.exception(
                    "Face alignment failed for face %d",
                    index,
                )

                continue

            if aligned_face is None:

                logger.warning(
                    "Alignment returned None for face %d",
                    index,
                )

                continue

            logger.info(
                "Aligned face shape: %s",
                aligned_face.shape,
            )

            # ==============================================================
            # STEP 2B — ARCFACE EMBEDDING
            # ==============================================================

            logger.info(
                "Generating ArcFace embedding..."
            )

            try:

                embedding = (
                    self.recognizer.get_embedding(
                        aligned_face
                    )
                )

            except Exception:

                logger.exception(
                    "Embedding generation failed for face %d",
                    index,
                )

                continue

            if embedding is None:

                logger.warning(
                    "Embedding returned None for face %d",
                    index,
                )

                continue

            embedding = np.asarray(
                embedding,
                dtype=np.float32,
            )

            logger.info(
                "Embedding shape: %s",
                embedding.shape,
            )

            embedding_norm = float(
                np.linalg.norm(
                    embedding
                )
            )

            logger.info(
                "Embedding norm: %.6f",
                embedding_norm,
            )

            # ==============================================================
            # VALIDATE EMBEDDING
            # ==============================================================

            if embedding.ndim != 1:

                logger.warning(
                    "Unexpected embedding dimensions "
                    "for face %d: %s",
                    index,
                    embedding.shape,
                )

                continue

            if embedding.shape[0] != 512:

                logger.warning(
                    "Unexpected embedding size "
                    "for face %d: %d",
                    index,
                    embedding.shape[0],
                )

                continue

            # ==============================================================
            # STEP 2C — AGE + GENDER
            # ==============================================================

            logger.info(
                "Running age/gender classification..."
            )

            try:

                attributes = (
                    self.classifier.predict(
                        aligned_face
                    )
                )

            except Exception:

                logger.exception(
                    "Age/gender classification failed "
                    "for face %d",
                    index,
                )

                continue

            if not attributes:

                logger.warning(
                    "Attribute classifier returned "
                    "empty result for face %d",
                    index,
                )

                continue

            # ==============================================================
            # EXTRACT GENDER
            # ==============================================================

            gender = attributes.get(
                "gender"
            )

            if not gender:

                logger.warning(
                    "Gender result missing "
                    "for face %d",
                    index,
                )

                continue

            gender_label = gender.get(
                "label"
            )

            gender_confidence = gender.get(
                "confidence"
            )

            if gender_label is None:

                logger.warning(
                    "Gender label missing "
                    "for face %d",
                    index,
                )

                continue

            if gender_confidence is None:

                logger.warning(
                    "Gender confidence missing "
                    "for face %d",
                    index,
                )

                continue

            gender_confidence = float(
                gender_confidence
            )

            # ==============================================================
            # EXTRACT AGE
            # ==============================================================

            age = attributes.get(
                "age"
            )

            if not age:

                logger.warning(
                    "Age result missing "
                    "for face %d",
                    index,
                )

                continue

            age_years = age.get(
                "years"
            )

            age_group = age.get(
                "group"
            )

            # --------------------------------------------------------------
            # Age confidence is OPTIONAL.
            #
            # The new model currently does not necessarily provide
            # a calibrated confidence value for estimated age.
            # Therefore we must NEVER do:
            #
            #     float(None)
            #
            # --------------------------------------------------------------

            age_confidence = age.get(
                "confidence"
            )

            if age_years is None:

                logger.warning(
                    "Age years missing "
                    "for face %d",
                    index,
                )

                continue

            if age_group is None:

                logger.warning(
                    "Age group missing "
                    "for face %d",
                    index,
                )

                continue

            age_years = int(
                age_years
            )

            age_group = str(
                age_group
            )

            if age_confidence is not None:

                age_confidence = float(
                    age_confidence
                )

            # ==============================================================
            # LOG ATTRIBUTES
            # ==============================================================

            logger.info(
                "Gender: %s (%.4f)",
                gender_label,
                gender_confidence,
            )

            if age_confidence is not None:

                logger.info(
                    "Age: %d years (%s) "
                    "(confidence %.4f)",
                    age_years,
                    age_group,
                    age_confidence,
                )

            else:

                logger.info(
                    "Age: %d years (%s)",
                    age_years,
                    age_group,
                )

            # ==============================================================
            # STEP 2D — CREATE FACE RECORD
            # ==============================================================

            record = FaceRecord(

                face_index=index,

                # ----------------------------------------------------------
                # Detection
                # ----------------------------------------------------------

                detection_confidence=float(
                    detection.score
                ),

                bbox=[
                    float(value)
                    for value in detection.bbox
                ],

                # ----------------------------------------------------------
                # ArcFace
                # ----------------------------------------------------------

                embedding=[
                    float(value)
                    for value in embedding
                ],

                # ----------------------------------------------------------
                # Gender
                # ----------------------------------------------------------

                gender=GenderResult(

                    label=str(
                        gender_label
                    ),

                    confidence=(
                        gender_confidence
                    ),
                ),

                # ----------------------------------------------------------
                # Age
                # ----------------------------------------------------------

                age=AgeResult(

                    years=age_years,

                    group=age_group,

                    confidence=(
                        age_confidence
                        if age_confidence is not None
                        else None
                    ),
                ),

                # ----------------------------------------------------------
                # Source image
                # ----------------------------------------------------------

                image_path=str(
                    image_path
                ),
            )

            records.append(
                record
            )

            logger.info(
                "Face %d completed successfully.",
                index,
            )

        # ==================================================================
        # FINAL SUMMARY
        # ==================================================================

        logger.info("=" * 80)
        logger.info("FACE PIPELINE COMPLETE")
        logger.info("=" * 80)

        logger.info(
            "Input faces      : %d",
            len(detections),
        )

        logger.info(
            "Successful faces : %d",
            len(records),
        )

        failed_faces = (
            len(detections)
            - len(records)
        )

        logger.info(
            "Failed faces     : %d",
            failed_faces,
        )

        logger.info("=" * 80)

        return records