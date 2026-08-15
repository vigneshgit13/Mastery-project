import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort


logger = logging.getLogger(__name__)


@dataclass
class FaceDetection:
    bbox: Tuple[float, float, float, float]
    score: float
    landmarks: np.ndarray


class SCRFDDetector:

    def __init__(
        self,
        model_path: str | Path,
        input_size: int = 640,
        confidence_threshold: float = 0.50,
        nms_threshold: float = 0.40,
    ):

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"SCRFD model not found: {self.model_path}"
            )

        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        logger.info(
            "Loading SCRFD model: %s",
            self.model_path,
        )

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name

        logger.info(
            "SCRFD input: %s",
            self.input_name,
        )

        logger.info(
            "SCRFD providers: %s",
            self.session.get_providers(),
        )

    # ================================================================
    # PREPROCESS
    # ================================================================

    def preprocess(self, image: np.ndarray):

        if image is None:
            raise ValueError("Input image is None")

        if image.ndim != 3:
            raise ValueError(
                f"Expected BGR image, got shape {image.shape}"
            )

        original_height, original_width = image.shape[:2]

        scale = min(
            self.input_size / original_width,
            self.input_size / original_height,
        )

        resized_width = int(
            round(original_width * scale)
        )

        resized_height = int(
            round(original_height * scale)
        )

        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )

        canvas = np.zeros(
            (
                self.input_size,
                self.input_size,
                3,
            ),
            dtype=np.uint8,
        )

        canvas[
            :resized_height,
            :resized_width,
        ] = resized

        canvas = cv2.cvtColor(
            canvas,
            cv2.COLOR_BGR2RGB,
        )

        tensor = canvas.astype(
            np.float32
        )

        tensor = (
            tensor - 127.5
        ) / 128.0

        tensor = np.transpose(
            tensor,
            (2, 0, 1),
        )

        tensor = np.expand_dims(
            tensor,
            axis=0,
        )

        return (
            tensor,
            scale,
            original_width,
            original_height,
        )

    # ================================================================
    # ANCHOR CENTERS
    # ================================================================

    @staticmethod
    def generate_centers(
        feature_size: int,
        stride: int,
        num_anchors: int = 2,
    ) -> np.ndarray:

        centers = []

        for y in range(feature_size):

            for x in range(feature_size):

                for _ in range(num_anchors):

                    centers.append(
                        [
                            x,
                            y,
                        ]
                    )

        return np.asarray(
            centers,
            dtype=np.float32
        )
    # ================================================================
    # BBOX DECODER
    # ================================================================

    @staticmethod
    def decode_bboxes(
        centers: np.ndarray,
        deltas: np.ndarray,
        stride: int,
    ) -> np.ndarray:

        centers = centers * stride

        x1 = (
            centers[:, 0]
            - deltas[:, 0] * stride
        )

        y1 = (
            centers[:, 1]
            - deltas[:, 1] * stride
        )

        x2 = (
            centers[:, 0]
            + deltas[:, 2] * stride
        )

        y2 = (
            centers[:, 1]
            + deltas[:, 3] * stride
        )

        return np.stack(
            [
                x1,
                y1,
                x2,
                y2,
            ],
            axis=1,
        )

    # ================================================================
    # LANDMARK DECODER
    # ================================================================

    @staticmethod
    def decode_landmarks(
        centers: np.ndarray,
        deltas: np.ndarray,
        stride: int,
    ) -> np.ndarray:

        centers = centers * stride

        landmarks = []

        for i in range(5):

            x = (
                centers[:, 0]
                + deltas[:, i * 2]
                * stride
            )

            y = (
                centers[:, 1]
                + deltas[:, i * 2 + 1]
                * stride
            )

            landmarks.append(
                np.stack(
                    [x, y],
                    axis=1,
                )
            )

        return np.stack(
            landmarks,
            axis=1,
        )

    # ================================================================
    # NMS
    # ================================================================

    @staticmethod
    def nms(
        boxes: np.ndarray,
        scores: np.ndarray,
        threshold: float,
    ) -> List[int]:

        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (
            np.maximum(
                0,
                x2 - x1,
            )
            *
            np.maximum(
                0,
                y2 - y1,
            )
        )

        order = scores.argsort()[::-1]

        keep = []

        while len(order) > 0:

            i = order[0]

            keep.append(
                int(i)
            )

            if len(order) == 1:
                break

            xx1 = np.maximum(
                x1[i],
                x1[order[1:]],
            )

            yy1 = np.maximum(
                y1[i],
                y1[order[1:]],
            )

            xx2 = np.minimum(
                x2[i],
                x2[order[1:]],
            )

            yy2 = np.minimum(
                y2[i],
                y2[order[1:]],
            )

            width = np.maximum(
                0,
                xx2 - xx1,
            )

            height = np.maximum(
                0,
                yy2 - yy1,
            )

            intersection = (
                width * height
            )

            union = (
                areas[i]
                + areas[order[1:]]
                - intersection
            )

            iou = np.zeros_like(
                intersection
            )

            valid = union > 0

            iou[valid] = (
                intersection[valid]
                / union[valid]
            )

            remaining = np.where(
                iou <= threshold
            )[0]

            order = order[
                remaining + 1
            ]

        return keep

    # ================================================================
    # MAIN DETECTION METHOD
    # ================================================================

    def detect(
        self,
        image: np.ndarray,
    ) -> List[FaceDetection]:

        tensor, scale, original_width, original_height = (
            self.preprocess(image)
        )

        logger.info(
            "Running SCRFD inference..."
        )

        outputs = self.session.run(
            None,
            {
                self.input_name: tensor,
            },
        )

        logger.info(
            "SCRFD returned %d outputs",
            len(outputs),
        )

        if len(outputs) != 9:
            raise RuntimeError(
                f"Expected 9 SCRFD outputs, "
                f"got {len(outputs)}"
            )

        # ------------------------------------------------------------
        # Your model output order
        #
        # 0,1,2 -> scores
        # 3,4,5 -> bbox
        # 6,7,8 -> landmarks
        # ------------------------------------------------------------

        score_outputs = [
            outputs[0],
            outputs[1],
            outputs[2],
        ]

        bbox_outputs = [
            outputs[3],
            outputs[4],
            outputs[5],
        ]

        landmark_outputs = [
            outputs[6],
            outputs[7],
            outputs[8],
        ]

        strides = [
            8,
            16,
            32,
        ]
        num_anchors = 2
        all_boxes = []
        all_scores = []
        all_landmarks = []

        # ------------------------------------------------------------
        # Decode each feature level
        # ------------------------------------------------------------

        for (
            score_output,
            bbox_output,
            landmark_output,
            stride,
        ) in zip(
            score_outputs,
            bbox_outputs,
            landmark_outputs,
            strides,
        ):

            scores = np.asarray(
                score_output
            ).reshape(-1)

            bbox_deltas = np.asarray(
                bbox_output
            ).reshape(-1, 4)

            landmark_deltas = np.asarray(
                landmark_output
            ).reshape(-1, 10)

            feature_size = (
                self.input_size
                // stride
            )

            num_anchors = 2

            expected = (
                feature_size
                *feature_size
                *num_anchors
            )

            expected = (
                feature_size
                * feature_size
                * num_anchors
            )

            logger.info(
                "Stride %d: scores=%d expected=%d",
                stride,
                len(scores),
                expected,
            )

            if len(scores) != expected:

                raise RuntimeError(
                    f"Unexpected output size "
                    f"for stride {stride}: "
                    f"{len(scores)} vs {expected}"
                )

            centers = self.generate_centers(
                feature_size,
                stride,
                num_anchors=2,
            )

            boxes = self.decode_bboxes(
                centers,
                bbox_deltas,
                stride,
            )

            landmarks = self.decode_landmarks(
                centers,
                landmark_deltas,
                stride,
            )

            # --------------------------------------------------------
            # Confidence filtering
            # --------------------------------------------------------

            mask = (
                scores
                >= self.confidence_threshold
            )

            if not np.any(mask):
                continue

            boxes = boxes[mask]

            scores_filtered = (
                scores[mask]
            )

            landmarks = landmarks[mask]

            all_boxes.append(
                boxes
            )

            all_scores.append(
                scores_filtered
            )

            all_landmarks.append(
                landmarks
            )

        # ------------------------------------------------------------
        # No faces
        # ------------------------------------------------------------

        if not all_boxes:

            logger.info(
                "No faces detected."
            )

            return []

        # ------------------------------------------------------------
        # Merge all scales
        # ------------------------------------------------------------

        boxes = np.concatenate(
            all_boxes,
            axis=0,
        )

        scores = np.concatenate(
            all_scores,
            axis=0,
        )

        landmarks = np.concatenate(
            all_landmarks,
            axis=0,
        )

        logger.info(
            "Candidates after confidence filtering: %d",
            len(boxes),
        )

        # ------------------------------------------------------------
        # Convert from model coordinates
        # to original image coordinates
        # ------------------------------------------------------------

        boxes /= scale

        landmarks /= scale

        # ------------------------------------------------------------
        # Clip bounding boxes
        # ------------------------------------------------------------

        boxes[:, 0] = np.clip(
            boxes[:, 0],
            0,
            original_width - 1,
        )

        boxes[:, 1] = np.clip(
            boxes[:, 1],
            0,
            original_height - 1,
        )

        boxes[:, 2] = np.clip(
            boxes[:, 2],
            0,
            original_width - 1,
        )

        boxes[:, 3] = np.clip(
            boxes[:, 3],
            0,
            original_height - 1,
        )

        # ------------------------------------------------------------
        # NMS
        # ------------------------------------------------------------

        keep = self.nms(
            boxes,
            scores,
            self.nms_threshold,
        )

        logger.info(
            "Candidates after NMS: %d",
            len(keep),
        )

        detections = []

        for index in keep:

            detection = FaceDetection(
                bbox=(
                    float(
                        boxes[index, 0]
                    ),
                    float(
                        boxes[index, 1]
                    ),
                    float(
                        boxes[index, 2]
                    ),
                    float(
                        boxes[index, 3]
                    ),
                ),
                score=float(
                    scores[index]
                ),
                landmarks=landmarks[index],
            )

            detections.append(
                detection
            )

        detections.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        logger.info(
            "Final faces detected: %d",
            len(detections),
        )

        return detections