import logging
from pathlib import Path

import cv2

from app.ai.detector import SCRFDDetector
from app.ai.aligner import face_aligner


logging.basicConfig(level=logging.INFO)


IMAGE_PATH = Path(
    "temp/b0ae1220-5117-4d73-8fae-766031dc8283.jpg"
)

OUTPUT_DIR = Path("temp/aligned")

MODEL_PATH = Path(
    "models/scrfd_10g_bnkps.onnx"
)


def main():

    print("=" * 80)
    print("FACE ALIGNMENT TEST")
    print("=" * 80)

    print(f"Image: {IMAGE_PATH}")

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise RuntimeError(
            f"Could not read image: {IMAGE_PATH}"
        )

    print(
        f"Original image shape: {image.shape}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # 1. Load detector
    # ---------------------------------------------------------

    detector = SCRFDDetector(
        model_path=MODEL_PATH,
    )

    # ---------------------------------------------------------
    # 2. Detect faces
    # ---------------------------------------------------------

    detections = detector.detect(image)

    print()
    print("=" * 80)
    print(
        f"DETECTED FACES: {len(detections)}"
    )
    print("=" * 80)

    # ---------------------------------------------------------
    # 3. Align faces
    # ---------------------------------------------------------

    aligned_faces = face_aligner.align_all(
        image,
        detections,
    )

    print()
    print("=" * 80)
    print(
        f"ALIGNED FACES: {len(aligned_faces)}"
    )
    print("=" * 80)

    # ---------------------------------------------------------
    # 4. Save aligned faces
    # ---------------------------------------------------------

    for index, aligned in enumerate(
        aligned_faces,
        start=1,
    ):

        output_path = (
            OUTPUT_DIR
            / f"face_{index:02d}.jpg"
        )

        success = cv2.imwrite(
            str(output_path),
            aligned,
        )

        if not success:
            raise RuntimeError(
                f"Failed to save: {output_path}"
            )

        print(
            f"Face {index}: "
            f"{aligned.shape} -> "
            f"{output_path}"
        )

    print()
    print("=" * 80)
    print("ALIGNMENT TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()