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

GROUP_IMAGE_DIR = Path(
    "temp/group_tests"
)

OUTPUT_DIR = Path(
    "temp/group_test_results"
)

SCRFD_MODEL_PATH = Path(
    "models/scrfd_10g_bnkps.onnx"
)

AGE_GENDER_MODEL_PATH = Path(
    "models/model.onnx"
)

PADDING_RATIO = 0.25

COLUMNS = 3

TILE_WIDTH = 320
TILE_HEIGHT = 380

HEADER_HEIGHT = 100


# ============================================================
# FACE CROP
# ============================================================

def crop_face_with_padding(
    image,
    bbox,
    padding_ratio=0.25,
):
    height, width = image.shape[:2]

    x1, y1, x2, y2 = bbox

    face_width = x2 - x1
    face_height = y2 - y1

    pad_x = face_width * padding_ratio
    pad_y = face_height * padding_ratio

    x1 = int(x1 - pad_x)
    y1 = int(y1 - pad_y)

    x2 = int(x2 + pad_x)
    y2 = int(y2 + pad_y)

    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(width, x2)
    y2 = min(height, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return image[y1:y2, x1:x2]


# ============================================================
# TEXT
# ============================================================

def draw_text(
    image,
    text,
    position,
    font_scale=0.65,
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
# CREATE CONTACT SHEET
# ============================================================

def create_contact_sheet(
    tiles,
    output_path,
):

    if not tiles:
        return

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

    sheet = np.zeros(
        (
            sheet_height,
            sheet_width,
            3,
        ),
        dtype=np.uint8,
    )

    for index, tile in enumerate(
        tiles
    ):

        row = index // COLUMNS
        col = index % COLUMNS

        y1 = row * TILE_HEIGHT
        y2 = y1 + TILE_HEIGHT

        x1 = col * TILE_WIDTH
        x2 = x1 + TILE_WIDTH

        sheet[
            y1:y2,
            x1:x2,
        ] = tile

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(output_path),
        sheet,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95,
        ],
    )


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    image_path,
    detector,
    classifier,
):

    print()
    print("=" * 80)
    print(
        f"IMAGE: {image_path.name}"
    )
    print("=" * 80)

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        logging.error(
            "Could not read image: %s",
            image_path,
        )
        return []

    print(
        f"Image shape: {image.shape}"
    )

    detections = detector.detect(
        image
    )

    print(
        f"Detected faces: "
        f"{len(detections)}"
    )

    tiles = []
    results = []

    for index, detection in enumerate(
        detections,
        start=1,
    ):

        face_crop = (
            crop_face_with_padding(
                image,
                detection.bbox,
                PADDING_RATIO,
            )
        )

        if (
            face_crop is None
            or face_crop.size == 0
        ):
            continue

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
            f"Face {index:02d} | "
            f"{gender_label:6s} "
            f"{gender_confidence:.3f} | "
            f"Age {estimated_age:3d} | "
            f"{age_group}"
        )

        # ----------------------------------------------------
        # Tile
        # ----------------------------------------------------

        face_display = cv2.resize(
            face_crop,
            (
                TILE_WIDTH,
                TILE_HEIGHT - HEADER_HEIGHT,
            ),
            interpolation=cv2.INTER_CUBIC,
        )

        tile = np.zeros(
            (
                TILE_HEIGHT,
                TILE_WIDTH,
                3,
            ),
            dtype=np.uint8,
        )

        # Header

        draw_text(
            tile,
            f"FACE {index:02d}",
            (15, 30),
            0.75,
            2,
        )

        draw_text(
            tile,
            (
                f"Gender: {gender_label} "
                f"({gender_confidence:.3f})"
            ),
            (15, 58),
            0.58,
            2,
        )

        draw_text(
            tile,
            (
                f"Age: {estimated_age} "
                f"years ({age_group})"
            ),
            (15, 86),
            0.58,
            2,
        )

        tile[
            HEADER_HEIGHT:
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

    return results, tiles


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("MULTIPLE GROUP IMAGE AGE/GENDER TEST")
    print("=" * 80)

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not GROUP_IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Directory not found: "
            f"{GROUP_IMAGE_DIR}"
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

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    image_files = sorted(
        [
            *GROUP_IMAGE_DIR.glob("*.jpg"),
            *GROUP_IMAGE_DIR.glob("*.jpeg"),
            *GROUP_IMAGE_DIR.glob("*.png"),
        ]
    )

    if not image_files:
        raise RuntimeError(
            f"No images found in "
            f"{GROUP_IMAGE_DIR}"
        )

    print()
    print(
        f"Found {len(image_files)} "
        f"group images."
    )

    # --------------------------------------------------------
    # Load models ONCE
    # --------------------------------------------------------

    print()
    print(
        "Loading SCRFD..."
    )

    detector = SCRFDDetector(
        model_path=str(
            SCRFD_MODEL_PATH
        )
    )

    print(
        "Loading age/gender model..."
    )

    classifier = GenderAgeClassifier(
        model_path=str(
            AGE_GENDER_MODEL_PATH
        )
    )

    # --------------------------------------------------------
    # Global statistics
    # --------------------------------------------------------

    total_faces = 0
    total_male = 0
    total_female = 0
    total_adult = 0
    total_child = 0

    all_ages = []

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    for image_path in image_files:

        result = process_image(
            image_path,
            detector,
            classifier,
        )

        if not result:
            continue

        results, tiles = result

        total_faces += len(results)

        total_male += sum(
            r["gender"] == "Male"
            for r in results
        )

        total_female += sum(
            r["gender"] == "Female"
            for r in results
        )

        total_adult += sum(
            r["age_group"] == "Adult"
            for r in results
        )

        total_child += sum(
            r["age_group"] == "Child"
            for r in results
        )

        all_ages.extend(
            r["age"]
            for r in results
        )

        # ----------------------------------------------------
        # Contact sheet
        # ----------------------------------------------------

        output_name = (
            image_path.stem
            + "_contact_sheet.jpg"
        )

        output_path = (
            OUTPUT_DIR
            / output_name
        )

        create_contact_sheet(
            tiles,
            output_path,
        )

        print(
            f"Contact sheet: "
            f"{output_path}"
        )

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)

    print(
        f"Images processed : "
        f"{len(image_files)}"
    )

    print(
        f"Total faces      : "
        f"{total_faces}"
    )

    print()
    print(
        f"Male             : "
        f"{total_male}"
    )

    print(
        f"Female           : "
        f"{total_female}"
    )

    print()
    print(
        f"Adult            : "
        f"{total_adult}"
    )

    print(
        f"Child            : "
        f"{total_child}"
    )

    if all_ages:

        print()
        print(
            f"Minimum age      : "
            f"{min(all_ages)}"
        )

        print(
            f"Maximum age      : "
            f"{max(all_ages)}"
        )

        print(
            f"Average age      : "
            f"{sum(all_ages) / len(all_ages):.1f}"
        )

    print()
    print(
        f"Results directory: "
        f"{OUTPUT_DIR}"
    )

    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()