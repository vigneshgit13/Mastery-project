import logging
from pathlib import Path

import cv2

from app.ai.detector import SCRFDDetector
from app.ai.attributes import GenderAgeClassifier


logging.basicConfig(
    level=logging.INFO,
)


IMAGE_PATH = Path(
    "temp/b0ae1220-5117-4d73-8fae-766031dc8283.jpg"
)

SCRFD_MODEL_PATH = Path(
    "models/scrfd_10g_bnkps.onnx"
)

AGE_GENDER_MODEL_PATH = Path(
    "models/model.onnx"
)

ATTRIBUTE_CROP_DIR = Path(
    "temp/attribute_crops_vit"
)

PADDING_RATIO = 0.25


def crop_face_with_padding(
    image,
    bbox,
    padding_ratio=0.25,
):
    """
    Crop the detected face from the ORIGINAL image.

    Adds padding around the SCRFD bounding box.
    """

    image_height, image_width = (
        image.shape[:2]
    )

    x1, y1, x2, y2 = bbox

    face_width = x2 - x1
    face_height = y2 - y1

    pad_x = (
        face_width * padding_ratio
    )

    pad_y = (
        face_height * padding_ratio
    )

    x1 = int(x1 - pad_x)
    y1 = int(y1 - pad_y)
    x2 = int(x2 + pad_x)
    y2 = int(y2 + pad_y)

    # Clamp to image
    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(image_width, x2)
    y2 = min(image_height, y2)

    crop = image[
        y1:y2,
        x1:x2,
    ]

    return crop


def main():

    print("=" * 80)
    print("ViT AGE + GENDER CLASSIFICATION TEST")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Validate files
    # ------------------------------------------------------------------

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Image not found: {IMAGE_PATH}"
        )

    if not SCRFD_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"SCRFD model not found: "
            f"{SCRFD_MODEL_PATH}"
        )

    if not AGE_GENDER_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: "
            f"{AGE_GENDER_MODEL_PATH}"
        )

    # ------------------------------------------------------------------
    # Load image
    # ------------------------------------------------------------------

    image = cv2.imread(
        str(IMAGE_PATH)
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image: "
            f"{IMAGE_PATH}"
        )

    print()
    print(
        f"Image: {IMAGE_PATH}"
    )

    print(
        f"Image shape: {image.shape}"
    )

    # ------------------------------------------------------------------
    # Load SCRFD
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("LOADING SCRFD")
    print("=" * 80)

    detector = SCRFDDetector(
        model_path=str(
            SCRFD_MODEL_PATH
        )
    )

    # ------------------------------------------------------------------
    # Detect
    # ------------------------------------------------------------------

    print()
    print(
        "Running face detection..."
    )

    detections = detector.detect(
        image
    )

    print()
    print(
        f"Detected faces: "
        f"{len(detections)}"
    )

    if not detections:
        raise RuntimeError(
            "No faces detected."
        )

    # ------------------------------------------------------------------
    # Load ViT age/gender model
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("LOADING ViT AGE + GENDER MODEL")
    print("=" * 80)

    classifier = GenderAgeClassifier(
        model_path=str(
            AGE_GENDER_MODEL_PATH
        )
    )

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------

    ATTRIBUTE_CROP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    # ------------------------------------------------------------------
    # Process faces
    # ------------------------------------------------------------------

    for index, detection in enumerate(
        detections,
        start=1,
    ):

        print()
        print("=" * 80)
        print(
            f"PROCESSING FACE "
            f"{index}/{len(detections)}"
        )
        print("=" * 80)

        bbox = detection.bbox
        score = detection.score

        print(
            f"SCRFD score: "
            f"{score:.6f}"
        )

        print(
            f"BBox: {bbox}"
        )

        # --------------------------------------------------------------
        # Attribute crop
        # --------------------------------------------------------------

        face_crop = (
            crop_face_with_padding(
                image,
                bbox,
                PADDING_RATIO,
            )
        )

        if (
            face_crop is None
            or face_crop.size == 0
        ):
            print(
                "ERROR: Empty crop."
            )
            continue

        print(
            f"Crop shape: "
            f"{face_crop.shape}"
        )

        # --------------------------------------------------------------
        # Save crop
        # --------------------------------------------------------------

        crop_path = (
            ATTRIBUTE_CROP_DIR
            / f"face_{index:02d}.jpg"
        )

        cv2.imwrite(
            str(crop_path),
            face_crop,
        )

        print(
            f"Saved crop: "
            f"{crop_path}"
        )

        # --------------------------------------------------------------
        # Predict
        # --------------------------------------------------------------

        result = classifier.predict(
            face_crop
        )

        gender = result[
            "gender"
        ]

        age = result[
            "age"
        ]

        # --------------------------------------------------------------
        # Gender
        # --------------------------------------------------------------

        print()
        print("GENDER")

        print(
            f"  Label              : "
            f"{gender['label']}"
        )

        print(
            f"  Confidence         : "
            f"{gender['confidence']:.6f}"
        )

        print(
            f"  Male probability   : "
            f"{gender['male_probability']:.6f}"
        )

        print(
            f"  Female probability : "
            f"{gender['female_probability']:.6f}"
        )

        # --------------------------------------------------------------
        # Age
        # --------------------------------------------------------------

        print()
        print("AGE")

        print(
            f"  Estimated age      : "
            f"{age['years']}"
        )

        print(
            f"  Age group          : "
            f"{age['group']}"
        )

        print(
            f"  Raw model age      : "
            f"{age['raw']:.4f}"
        )

        # --------------------------------------------------------------
        # Store
        # --------------------------------------------------------------

        results.append(
            {
                "face": index,
                "gender": gender["label"],
                "gender_confidence": (
                    gender["confidence"]
                ),
                "age": age["years"],
                "age_group": age["group"],
                "age_raw": age["raw"],
            }
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for result in results:

        print(
            f"Face {result['face']:02d} | "
            f"Gender: "
            f"{result['gender']:6s} "
            f"({result['gender_confidence']:.4f}) | "
            f"Age: "
            f"{result['age']:3d} | "
            f"Group: "
            f"{result['age_group']}"
        )

    # ------------------------------------------------------------------
    # Gender counts
    # ------------------------------------------------------------------

    male_count = sum(
        1
        for result in results
        if result["gender"] == "Male"
    )

    female_count = sum(
        1
        for result in results
        if result["gender"] == "Female"
    )

    print()
    print("=" * 80)
    print("GENDER COUNTS")
    print("=" * 80)

    print(
        f"Male  : {male_count}"
    )

    print(
        f"Female: {female_count}"
    )

    # ------------------------------------------------------------------
    # Age group counts
    # ------------------------------------------------------------------

    child_count = sum(
        1
        for result in results
        if result["age_group"] == "Child"
    )

    adult_count = sum(
        1
        for result in results
        if result["age_group"] == "Adult"
    )

    print()
    print("=" * 80)
    print("AGE GROUP COUNTS")
    print("=" * 80)

    print(
        f"Child : {child_count}"
    )

    print(
        f"Adult : {adult_count}"
    )

    print()
    print("=" * 80)
    print("ViT AGE + GENDER TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()