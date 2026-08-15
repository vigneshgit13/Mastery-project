import logging
from typing import Tuple

import cv2
import numpy as np

from app.ai.detector import FaceDetection


logger = logging.getLogger(__name__)


class FaceAligner:
    """
    Aligns detected faces using SCRFD's 5-point facial landmarks.

    Output:
        112 x 112 aligned face suitable for ArcFace.
    """

    # Standard ArcFace 112x112 reference landmarks.
    REFERENCE_LANDMARKS = np.array(
        [
            [38.2946, 51.6963],  # left eye
            [73.5318, 51.5014],  # right eye
            [56.0252, 71.7366],  # nose
            [41.5493, 92.3655],  # left mouth
            [70.7299, 92.2041],  # right mouth
        ],
        dtype=np.float32,
    )

    OUTPUT_SIZE: Tuple[int, int] = (112, 112)

    def __init__(self):
        logger.info("FaceAligner initialized.")

    def align(
        self,
        image: np.ndarray,
        detection: FaceDetection,
    ) -> np.ndarray:
        """
        Align one detected face.

        Args:
            image:
                Original BGR image.

            detection:
                FaceDetection containing 5-point landmarks.

        Returns:
            Aligned 112x112 BGR face.
        """

        if image is None:
            raise ValueError("Image is None.")

        if image.size == 0:
            raise ValueError("Image is empty.")

        landmarks = np.asarray(
            detection.landmarks,
            dtype=np.float32,
        )

        if landmarks.shape != (5, 2):
            raise ValueError(
                f"Expected landmarks shape (5, 2), "
                f"got {landmarks.shape}"
            )

        # Estimate similarity transformation.
        transform_matrix, _ = cv2.estimateAffinePartial2D(
            landmarks,
            self.REFERENCE_LANDMARKS,
            method=cv2.LMEDS,
        )

        if transform_matrix is None:
            raise RuntimeError(
                "Could not calculate face alignment transform."
            )

        aligned = cv2.warpAffine(
            image,
            transform_matrix,
            self.OUTPUT_SIZE,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        if aligned.shape[:2] != (112, 112):
            raise RuntimeError(
                f"Unexpected aligned face shape: {aligned.shape}"
            )

        return aligned

    def align_all(
        self,
        image: np.ndarray,
        detections: list[FaceDetection],
    ) -> list[np.ndarray]:
        """
        Align all detected faces.
        """

        aligned_faces = []

        for index, detection in enumerate(detections, start=1):

            try:
                aligned = self.align(
                    image,
                    detection,
                )

                aligned_faces.append(aligned)

                logger.info(
                    "Aligned face %d/%d",
                    index,
                    len(detections),
                )

            except Exception:
                logger.exception(
                    "Failed to align face %d.",
                    index,
                )

        logger.info(
            "Successfully aligned %d/%d faces.",
            len(aligned_faces),
            len(detections),
        )

        return aligned_faces


face_aligner = FaceAligner()