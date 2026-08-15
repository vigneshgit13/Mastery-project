import logging
from pathlib import Path

import cv2

from app.ai.detector import SCRFDDetector


logging.basicConfig(
    level=logging.INFO,
)


BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "scrfd_10g_bnkps.onnx"
)

IMAGE_PATH = (
    BASE_DIR
    / "temp"
    / "b0ae1220-5117-4d73-8fae-766031dc8283.jpg"
)

OUTPUT_PATH = (
    BASE_DIR
    / "temp"
    / "detector_test.jpg"
)


def main():

    print("=" * 80)
    print("SCRFD DETECTION TEST")
    print("=" * 80)

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Test image not found: {IMAGE_PATH}"
        )

    image = cv2.imread(
        str(IMAGE_PATH)
    )

    if image is None:
        raise RuntimeError(
            f"OpenCV could not read: {IMAGE_PATH}"
        )

    print(
        "Image:",
        IMAGE_PATH,
    )

    print(
        "Image shape:",
        image.shape,
    )

    detector = SCRFDDetector(
        model_path=MODEL_PATH,
        input_size=640,
        confidence_threshold=0.50,
        nms_threshold=0.40,
    )

    detections = detector.detect(
        image
    )

    print()
    print("=" * 80)
    print(
        f"DETECTED FACES: {len(detections)}"
    )
    print("=" * 80)

    for index, face in enumerate(
        detections,
        start=1,
    ):

        print(
            f"Face {index}"
        )

        print(
            "  Score:",
            face.score,
        )

        print(
            "  BBox:",
            face.bbox,
        )

        print(
            "  Landmarks:"
        )

        print(
            face.landmarks
        )

        print()

        x1, y1, x2, y2 = map(
            int,
            face.bbox,
        )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        for point in face.landmarks:

            x, y = map(
                int,
                point,
            )

            cv2.circle(
                image,
                (x, y),
                3,
                (0, 0, 255),
                -1,
            )

        cv2.putText(
            image,
            f"{face.score:.2f}",
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(
        str(OUTPUT_PATH),
        image,
    )

    print(
        "Output saved:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()