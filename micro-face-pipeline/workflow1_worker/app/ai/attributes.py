import logging
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import onnxruntime as ort


logger = logging.getLogger(__name__)


class GenderAgeClassifier:
    """
    Gender + age classifier using:

        onnx-community/age-gender-prediction-ONNX

    Model input:
        1 x 3 x 224 x 224

    Model output:
        logits[0] = predicted age
        logits[1] = female probability

    The model expects RGB images normalized using
    ImageNet mean/std.
    """

    def __init__(
        self,
        model_path: str,
    ):

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Age/gender model not found: "
                f"{self.model_path}"
            )

        logger.info(
            "Loading ViT age/gender model: %s",
            self.model_path,
        )

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=[
                "CPUExecutionProvider"
            ],
        )

        self.input_name = (
            self.session
            .get_inputs()[0]
            .name
        )

        self.output_names = [
            output.name
            for output in self.session.get_outputs()
        ]

        logger.info(
            "Model input: %s",
            self.input_name,
        )

        logger.info(
            "Model outputs: %s",
            self.output_names,
        )

        logger.info(
            "Providers: %s",
            self.session.get_providers(),
        )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def preprocess(
        face: np.ndarray,
    ) -> np.ndarray:
        """
        OpenCV BGR image -> model tensor.

        Steps:

            BGR
              ↓
            RGB
              ↓
            resize 224x224
              ↓
            float32 / 255
              ↓
            ImageNet normalization
              ↓
            HWC -> CHW
              ↓
            batch dimension
        """

        if face is None:
            raise ValueError(
                "Face image is None."
            )

        if face.size == 0:
            raise ValueError(
                "Face image is empty."
            )

        # --------------------------------------------------------------
        # BGR -> RGB
        # --------------------------------------------------------------

        image = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB,
        )

        # --------------------------------------------------------------
        # Resize
        # --------------------------------------------------------------

        image = cv2.resize(
            image,
            (224, 224),
            interpolation=cv2.INTER_LINEAR,
        )

        # --------------------------------------------------------------
        # uint8 -> float32
        # --------------------------------------------------------------

        image = image.astype(
            np.float32
        )

        # --------------------------------------------------------------
        # Scale [0,255] -> [0,1]
        # --------------------------------------------------------------

        image /= 255.0

        # --------------------------------------------------------------
        # ImageNet normalization
        # --------------------------------------------------------------

        mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32,
        )

        std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32,
        )

        image = (
            image - mean
        ) / std

        # --------------------------------------------------------------
        # HWC -> CHW
        # --------------------------------------------------------------

        image = np.transpose(
            image,
            (2, 0, 1),
        )

        # --------------------------------------------------------------
        # Add batch dimension
        # --------------------------------------------------------------

        image = np.expand_dims(
            image,
            axis=0,
        )

        return image.astype(
            np.float32
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        face: np.ndarray,
    ) -> Dict:

        tensor = self.preprocess(
            face
        )

        outputs = self.session.run(
            None,
            {
                self.input_name: tensor
            },
        )

        raw = np.asarray(
            outputs[0]
        )

        logger.debug(
            "Raw model output: %s",
            raw,
        )

        if raw.ndim != 2:
            raise RuntimeError(
                f"Unexpected output shape: "
                f"{raw.shape}"
            )

        if raw.shape[1] != 2:
            raise RuntimeError(
                f"Expected 2 outputs, "
                f"got: {raw.shape}"
            )

        # --------------------------------------------------------------
        # Model output
        #
        # [0] = age
        # [1] = female probability
        # --------------------------------------------------------------

        age_raw = float(
            raw[0, 0]
        )

        female_probability = float(
            raw[0, 1]
        )

        # --------------------------------------------------------------
        # Age
        # --------------------------------------------------------------

        estimated_age = int(
            np.clip(
                np.rint(age_raw),
                0,
                100,
            )
        )

        # --------------------------------------------------------------
        # Gender
        # --------------------------------------------------------------

        female_probability = float(
            np.clip(
                female_probability,
                0.0,
                1.0,
            )
        )

        male_probability = (
            1.0 - female_probability
        )

        if female_probability >= 0.5:

            gender_label = "Female"

            gender_confidence = (
                female_probability
            )

            gender_class_id = 1

        else:

            gender_label = "Male"

            gender_confidence = (
                male_probability
            )

            gender_class_id = 0

        # --------------------------------------------------------------
        # Age group
        # --------------------------------------------------------------

        if estimated_age < 18:

            age_group = "Child"

        else:

            age_group = "Adult"

        return {
            "gender": {
                "label": gender_label,
                "class_id": gender_class_id,
                "confidence": gender_confidence,
                "female_probability": (
                    female_probability
                ),
                "male_probability": (
                    male_probability
                ),
            },

            "age": {
                "years": estimated_age,
                "group": age_group,
                "raw": age_raw,
            },
        }