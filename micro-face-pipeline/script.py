import logging
from pathlib import Path

import cv2
import numpy as np

from app.ai.detector import SCRFDDetector
from app.ai.attributes import GenderAgeClassifier


logging.basicConfig(
    level=logging.INFO,
)


# ============================================================
# CONFIG
# ============================================================

IMAGE_PATH = Path(
    "temp/b0ae1220-5117-4d73-8fae-766031dc8283.jpg"
)

SCRFD_MODEL_PATH = Path(
    "models/scrfd_10g_bnkps.onnx"
)

AGE_GENDER_MODEL_PATH = Path(
    "models/model.onnx"
)

OUTPUT_PATH = Path(
    "temp/contact_sheet_vit_age_gender.jpg"
)

# Padding around detected face before sending to model
PADDING_RATIO = 0.25

# Contact sheet layout
COLUMNS = 3

# Individual face tile size
TILE_WIDTH = 320
TILE_HEIGHT = 380

# Header height
HEADER_HEIGHT = 100


# ============================================================
# FACE CROP
# ============================================================

def crop_face_with_padding(
    image,
    bbox,
    padding_ratio=0.25,
):
    """
    Crop face from original image with padding.
    """

    image_height, image_width = image.shape[:2]

    x1, y1, x2, y2 = bbox

    face_width = x2 - x1
    face_height = y2 - y1

    pad_x = face_width * padding_ratio
    pad_y = face_height * padding_ratio

    x1 = int(x1 - pad_x)
    y1 = int(y1 - pad_y)

    x2 = int(x2 + pad_x)
    y2 = int(y2 + pad_y)

    # Clamp coordinates
    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(image_width, x2)
    y2 = min(image_height, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[
        y1:y2,
        x1:x2,
    ]

    return crop


# ============================================================
# TEXT HELPERS
# ============================================================

def draw_text(
    image,
    text,
    position,
    font_scale=0.7,
    thickness=2,
):
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("NEW AGE/GENDER MODEL CONTACT SHEET")
    print("=" * 80)

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

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
            f"Age/gender model not found: "
            f"{AGE_GENDER_MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Load original image
    # --------------------------------------------------------

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
        f"Original image: {IMAGE_PATH}"
    )

    print(
        f"Image shape: {image.shape}"
    )

    # --------------------------------------------------------
    # Load SCRFD
    # --------------------------------------------------------

    print()
    print("Loading SCRFD...")

    detector = SCRFDDetector(
        model_path=str(
            SCRFD_MODEL_PATH
        )
    )

    # --------------------------------------------------------
    # Detect faces
    # --------------------------------------------------------

    print()
    print("Detecting faces...")

    detections = detector.detect(
        image
    )

    print(
        f"Detected faces: "
        f"{len(detections)}"
    )

    if not detections:
        raise RuntimeError(
            "No faces detected."
        )

    # --------------------------------------------------------
    # Load new model
    # --------------------------------------------------------

    print()
    print("Loading new age/gender model...")

    classifier = GenderAgeClassifier(
        model_path=str(
            AGE_GENDER_MODEL_PATH
        )
    )

    # --------------------------------------------------------
    # Process faces
    # --------------------------------------------------------

    tiles = []
    results = []

    for index, detection in enumerate(
        detections,
        start=1,
    ):

        print()
        print(
            f"Processing face "
            f"{index}/{len(detections)}"
        )

        # ----------------------------------------------------
        # Crop
        # ----------------------------------------------------

        face_crop = crop_face_with_padding(
            image,
            detection.bbox,
            PADDING_RATIO,
        )

        if (
            face_crop is None
            or face_crop.size == 0
        ):
            logging.warning(
                "Skipping face %d: empty crop",
                index,
            )
            continue

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        result = classifier.predict(
            face_crop
        )

        gender = result["gender"]
        age = result["age"]

        gender_label = gender["label"]
        gender_confidence = (
            gender["confidence"]
        )

        estimated_age = age["years"]
        age_group = age["group"]

        print(
            f"Gender: {gender_label} "
            f"({gender_confidence:.4f})"
        )

        print(
            f"Age: {estimated_age} "
            f"({age_group})"
        )

        # ----------------------------------------------------
        # Resize face
        # ----------------------------------------------------

        face_display = cv2.resize(
            face_crop,
            (
                TILE_WIDTH,
                TILE_HEIGHT - HEADER_HEIGHT,
            ),
            interpolation=cv2.INTER_CUBIC,
        )

        # ----------------------------------------------------
        # Create tile
        # ----------------------------------------------------

        tile = np.zeros(
            (
                TILE_HEIGHT,
                TILE_WIDTH,
                3,
            ),
            dtype=np.uint8,
        )

        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        header = tile[
            :HEADER_HEIGHT,
            :,
        ]

        # Face number
        draw_text(
            header,
            f"FACE {index:02d}",
            (15, 30),
            font_scale=0.75,
            thickness=2,
        )

        # Gender
        draw_text(
            header,
            (
                f"Gender: {gender_label} "
                f"({gender_confidence:.3f})"
            ),
            (15, 58),
            font_scale=0.58,
            thickness=2,
        )

        # Age
        draw_text(
            header,
            (
                f"Age: {estimated_age} "
                f"years ({age_group})"
            ),
            (15, 86),
            font_scale=0.58,
            thickness=2,
        )

        # ----------------------------------------------------
        # Put face below header
        # ----------------------------------------------------

        tile[
            HEADER_HEIGHT:,
            :,
        ] = face_display

        tiles.append(tile)

        results.append(
            {
                "face": index,
                "gender": gender_label,
                "gender_confidence": (
                    gender_confidence
                ),
                "age": estimated_age,
                "age_group": age_group,
            }
        )

    # --------------------------------------------------------
    # Create contact sheet
    # --------------------------------------------------------

    if not tiles:
        raise RuntimeError(
            "No valid face tiles created."
        )

    rows = int(
        np.ceil(
            len(tiles) / COLUMNS
        )
    )

    sheet_width = (
        COLUMNS * TILE_WIDTH
    )

    sheet_height = (
        rows * TILE_HEIGHT
    )

    contact_sheet = np.zeros(
        (
            sheet_height,
            sheet_width,
            3,
        ),
        dtype=np.uint8,
    )

    # --------------------------------------------------------
    # Place tiles
    # --------------------------------------------------------

    for i, tile in enumerate(tiles):

        row = i // COLUMNS
        col = i % COLUMNS

        y1 = row * TILE_HEIGHT
        y2 = y1 + TILE_HEIGHT

        x1 = col * TILE_WIDTH
        x2 = x1 + TILE_WIDTH

        contact_sheet[
            y1:y2,
            x1:x2,
        ] = tile

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = cv2.imwrite(
        str(OUTPUT_PATH),
        contact_sheet,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95,
        ],
    )

    if not success:
        raise RuntimeError(
            f"Failed to save contact sheet: "
            f"{OUTPUT_PATH}"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("CONTACT SHEET COMPLETE")
    print("=" * 80)

    print(
        f"Faces processed: {len(results)}"
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )

    print()

    for result in results:

        print(
            f"Face {result['face']:02d} | "
            f"Gender: {result['gender']:6s} "
            f"({result['gender_confidence']:.4f}) | "
            f"Age: {result['age']:3d} | "
            f"{result['age_group']}"
        )

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()