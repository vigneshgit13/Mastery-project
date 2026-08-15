from pathlib import Path

import onnxruntime as ort


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"


class ModelLoader:

    def __init__(self):

        self.providers = [
            "CPUExecutionProvider",
        ]

        self.detector = None
        self.embedding = None
        self.gender_age = None

    def load_detector(self):

        model_path = MODEL_DIR / "scrfd_10g_bnkps.onnx"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Detector model not found: {model_path}"
            )

        self.detector = ort.InferenceSession(
            str(model_path),
            providers=self.providers,
        )

        return self.detector


model_loader = ModelLoader()