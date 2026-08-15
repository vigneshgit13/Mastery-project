import cv2
import numpy as np
from pathlib import Path

from app.ai.attributes import GenderAgeClassifier


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

CROP_DIR = Path("temp/attribute_crops")

OUTPUT_DIR = Path("temp/attribute_inspection")

GENDER_MODEL = Path(
    "models/gender_googlenet.onnx"
)

AGE_MODEL = Path(
    "models/age_googlenet.onnx"
)


# ----------------------------------------------------------------------
# Age labels
# ----------------------------------------------------------------------

AGE_LABELS = [
    "0-2",
    "4-6",
    "8-12",
    "15-20",
    "25-32",
    "38-43",
    "48-53",
    "60-100",
]


# ----------------------------------------------------------------------
# Draw result on image
# ----------------------------------------------------------------------

def draw_result(
    image,
    face_number,
    gender,
    age,
):
    """

    Draw prediction information on top of
    the individual face crop.

    """

    output = image.copy()

    # --------------------------------------------------------------
    # Resize for easier inspection
    # --------------------------------------------------------------

    output = cv2.resize(
        output,
        None,
        fx=3.0,
        fy=3.0,
        interpolation=cv2.INTER_NEAREST,
    )

    h, w = output.shape[:2]

    # --------------------------------------------------------------
    # Background panel
    # --------------------------------------------------------------

    panel_height = 120

    panel = np.zeros(
        (panel_height, w, 3),
        dtype=np.uint8,
    )

    output = np.vstack(
        [panel, output]
    )

    # --------------------------------------------------------------
    # Text
    # --------------------------------------------------------------

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(
        output,
        f"FACE {face_number:02d}",
        (10, 30),
        font,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        output,
        (
            f"Gender: {gender['label']} "
            f"({gender['confidence']:.4f})"
        ),
        (10, 65),
        font,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        output,
        (
            f"Age: {age['label']} "
            f"({age['confidence']:.4f})"
        ),
        (10, 100),
        font,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return output


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():

    print("=" * 80)
    print("ATTRIBUTE VISUAL INSPECTION")
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    classifier = GenderAgeClassifier(
        gender_model_path=str(
            GENDER_MODEL
        ),
        age_model_path=str(
            AGE_MODEL
        ),
    )

    face_files = sorted(
        CROP_DIR.glob("face_*.jpg")
    )

    if not face_files:
        raise RuntimeError(
            f"No crops found in {CROP_DIR}"
        )

    results = []

    for index, face_path in enumerate(
        face_files,
        start=1,
    ):

        print()
        print("=" * 80)
        print(
            f"FACE {index:02d}"
        )
        print("=" * 80)

        image = cv2.imread(
            str(face_path)
        )

        if image is None:
            print(
                f"Could not read {face_path}"
            )
            continue

        # ----------------------------------------------------------
        # Prediction
        # ----------------------------------------------------------

        result = classifier.predict(
            image
        )

        gender = result["gender"]
        age = result["age"]

        # ----------------------------------------------------------
        # Print prediction
        # ----------------------------------------------------------

        print(
            f"Gender : "
            f"{gender['label']} "
            f"({gender['confidence']:.4f})"
        )

        print(
            f"Age    : "
            f"{age['label']} "
            f"({age['confidence']:.4f})"
        )

        # ----------------------------------------------------------
        # Print ALL age scores
        # ----------------------------------------------------------

        print()
        print("AGE SCORES:")

        for class_id, score in enumerate(
            age["probabilities"]
        ):

            label = AGE_LABELS[
                class_id
            ]

            print(
                f"  {class_id}: "
                f"{label:>6} -> "
                f"{score:.6f}"
            )

        # ----------------------------------------------------------
        # Save annotated image
        # ----------------------------------------------------------

        annotated = draw_result(
            image=image,
            face_number=index,
            gender=gender,
            age=age,
        )

        output_path = (
            OUTPUT_DIR
            / f"face_{index:02d}_inspection.jpg"
        )

        cv2.imwrite(
            str(output_path),
            annotated,
        )

        print(
            f"Saved: {output_path}"
        )

        results.append(
            {
                "face": index,
                "gender": gender,
                "age": age,
            }
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL DIAGNOSTIC SUMMARY")
    print("=" * 80)

    for result in results:

        gender = result["gender"]
        age = result["age"]

        print(
            f"Face {result['face']:02d} | "
            f"Gender: {gender['label']:6s} "
            f"({gender['confidence']:.4f}) | "
            f"Age: {age['label']:6s} "
            f"({age['confidence']:.4f})"
        )

    print()
    print("=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)

    print()
    print(
        f"Inspection images: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()