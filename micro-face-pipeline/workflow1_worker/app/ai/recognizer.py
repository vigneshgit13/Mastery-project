import logging
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


logger = logging.getLogger(__name__)


class ArcFaceRecognizer:
    """
    ArcFace / GlintR100 face embedding extractor.

    Input:
        Aligned BGR face of shape (112, 112, 3)

    Output:
        L2-normalized 512-dimensional embedding.
    """

    def __init__(
        self,
        model_path: str | Path,
    ):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ArcFace model not found: {self.model_path}"
            )

        logger.info(
            "Loading ArcFace model: %s",
            self.model_path,
        )

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )

        inputs = self.session.get_inputs()

        if len(inputs) != 1:
            raise RuntimeError(
                f"Expected 1 input, found {len(inputs)}"
            )

        self.input_name = inputs[0].name
        self.input_shape = inputs[0].shape

        outputs = self.session.get_outputs()

        if len(outputs) != 1:
            raise RuntimeError(
                f"Expected 1 output, found {len(outputs)}"
            )

        self.output_name = outputs[0].name
        self.output_shape = outputs[0].shape

        logger.info(
            "ArcFace input: %s %s",
            self.input_name,
            self.input_shape,
        )

        logger.info(
            "ArcFace output: %s %s",
            self.output_name,
            self.output_shape,
        )

        providers = self.session.get_providers()

        logger.info(
            "ArcFace providers: %s",
            providers,
        )

    # ================================================================
    # PREPROCESS
    # ================================================================

    @staticmethod
    def preprocess(
        face: np.ndarray,
    ) -> np.ndarray:
        """
        Convert aligned BGR face into ArcFace input tensor.

        Input:
            (112, 112, 3) BGR uint8

        Output:
            (1, 3, 112, 112) float32
        """

        if face is None:
            raise ValueError(
                "Face image is None."
            )

        if face.size == 0:
            raise ValueError(
                "Face image is empty."
            )

        if face.shape[:2] != (112, 112):
            raise ValueError(
                f"Expected face size (112, 112), "
                f"got {face.shape[:2]}"
            )

        if face.ndim != 3:
            raise ValueError(
                f"Expected 3-channel image, "
                f"got shape {face.shape}"
            )

        # ------------------------------------------------------------
        # OpenCV image is BGR.
        # ArcFace expects RGB.
        # ------------------------------------------------------------

        rgb = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB,
        )

        # ------------------------------------------------------------
        # uint8 [0,255] -> float32
        #
        # Standard ArcFace normalization:
        #
        # (pixel - 127.5) / 127.5
        #
        # resulting approximately in [-1, 1]
        # ------------------------------------------------------------

        rgb = rgb.astype(
            np.float32
        )

        rgb = (
            rgb - 127.5
        ) / 127.5

        # ------------------------------------------------------------
        # HWC -> CHW
        # ------------------------------------------------------------

        chw = np.transpose(
            rgb,
            (2, 0, 1),
        )

        # ------------------------------------------------------------
        # CHW -> NCHW
        # ------------------------------------------------------------

        tensor = np.expand_dims(
            chw,
            axis=0,
        )

        return np.ascontiguousarray(
            tensor,
            dtype=np.float32,
        )

    # ================================================================
    # EMBEDDING
    # ================================================================

    def get_embedding(
        self,
        face: np.ndarray,
    ) -> np.ndarray:
        """
        Extract a normalized 512-D ArcFace embedding.
        """

        tensor = self.preprocess(
            face
        )

        output = self.session.run(
            [self.output_name],
            {
                self.input_name: tensor,
            },
        )[0]

        embedding = np.asarray(
            output,
            dtype=np.float32,
        ).reshape(-1)

        if embedding.shape[0] != 512:
            raise RuntimeError(
                "Unexpected ArcFace embedding "
                f"dimension: {embedding.shape[0]}"
            )

        # ------------------------------------------------------------
        # L2 normalization
        # ------------------------------------------------------------

        norm = np.linalg.norm(
            embedding
        )

        if norm == 0:
            raise RuntimeError(
                "ArcFace returned a zero embedding."
            )

        embedding = (
            embedding / norm
        )

        return embedding.astype(
            np.float32
        )

    # ================================================================
    # BATCH EMBEDDINGS
    # ================================================================

    def get_embeddings(
        self,
        faces: list[np.ndarray],
    ) -> list[np.ndarray]:
        """
        Generate embeddings for multiple aligned faces.
        """

        embeddings = []

        for index, face in enumerate(
            faces,
            start=1,
        ):

            try:

                embedding = self.get_embedding(
                    face
                )

                embeddings.append(
                    embedding
                )

                logger.info(
                    "Generated embedding "
                    "%d/%d",
                    index,
                    len(faces),
                )

            except Exception:

                logger.exception(
                    "Failed to generate "
                    "embedding for face %d.",
                    index,
                )

        logger.info(
            "Generated %d/%d embeddings.",
            len(embeddings),
            len(faces),
        )

        return embeddings


recognizer = ArcFaceRecognizer(
    model_path="models/glintr100.onnx"
)