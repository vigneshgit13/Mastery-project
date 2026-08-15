from pathlib import Path

from app.ai.detector import SCRFDDetector
from app.ai.aligner import FaceAligner
from app.ai.recognizer import ArcFaceRecognizer
from app.ai.attributes import GenderAgeClassifier

from app.ai.pipeline import FacePipeline


# ==============================================================
# PROJECT PATHS
# ==============================================================

BASE_DIR = Path(
    __file__
).resolve().parents[2]

MODELS_DIR = BASE_DIR / "models"


# ==============================================================
# MODEL PATHS
# ==============================================================

SCRFD_MODEL = (
    MODELS_DIR /
    "scrfd_10g_bnkps.onnx"
)

ARCFACE_MODEL = (
    MODELS_DIR /
    "glintr100.onnx"
)

AGE_GENDER_MODEL = (
    MODELS_DIR /
    "model.onnx"
)


def create_face_pipeline() -> FacePipeline:

    print("=" * 80)
    print("CREATING FACE PIPELINE")
    print("=" * 80)

    print(
        f"SCRFD model       : {SCRFD_MODEL}"
    )

    print(
        f"ArcFace model     : {ARCFACE_MODEL}"
    )

    print(
        f"Age/Gender model  : {AGE_GENDER_MODEL}"
    )

    # ----------------------------------------------------------
    # Verify models exist
    # ----------------------------------------------------------

    for model_path in [
        SCRFD_MODEL,
        ARCFACE_MODEL,
        AGE_GENDER_MODEL,
    ]:

        if not model_path.exists():

            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

    # ----------------------------------------------------------
    # Load detector
    # ----------------------------------------------------------

    detector = SCRFDDetector(
        model_path=SCRFD_MODEL,
        input_size=640,
        confidence_threshold=0.5,
        nms_threshold=0.4,
    )

    # ----------------------------------------------------------
    # Load aligner
    # ----------------------------------------------------------

    aligner = FaceAligner()

    # ----------------------------------------------------------
    # Load ArcFace
    # ----------------------------------------------------------

    recognizer = ArcFaceRecognizer(
        model_path=ARCFACE_MODEL
    )

    # ----------------------------------------------------------
    # Load Age/Gender model
    # ----------------------------------------------------------

    classifier = GenderAgeClassifier(
        model_path=str(
            AGE_GENDER_MODEL
        )
    )

    # ----------------------------------------------------------
    # Build pipeline
    # ----------------------------------------------------------

    pipeline = FacePipeline(
        detector=detector,
        aligner=aligner,
        recognizer=recognizer,
        classifier=classifier,
    )

    print("=" * 80)
    print("FACE PIPELINE READY")
    print("=" * 80)

    return pipeline