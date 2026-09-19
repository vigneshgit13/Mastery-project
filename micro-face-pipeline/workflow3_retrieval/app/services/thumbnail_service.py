from __future__ import annotations

from pathlib import Path

import cv2


class ThumbnailService:
    def __init__(self, repository) -> None:
        self.repository = repository

    def get_face_thumbnail(self, face_id: int) -> bytes | None:
        face = self.repository.get_face_thumbnail_data(face_id)

        if face is None:
            return None

        image_path = Path(face["local_path"])

        if not image_path.exists():
            raise FileNotFoundError(
                f"Source image not found: {image_path}"
            )

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(
                f"Unable to read source image: {image_path}"
            )

        height, width = image.shape[:2]

        x1 = max(0, int(face["bbox_x1"]))
        y1 = max(0, int(face["bbox_y1"]))
        x2 = min(width, int(face["bbox_x2"]))
        y2 = min(height, int(face["bbox_y2"]))

        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"Invalid bounding box for face {face_id}"
            )

        # Add 20% padding around the detected face.
        box_width = x2 - x1
        box_height = y2 - y1

        pad_x = int(box_width * 0.20)
        pad_y = int(box_height * 0.20)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(width, x2 + pad_x)
        y2 = min(height, y2 + pad_y)

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            raise ValueError(
                f"Empty thumbnail crop for face {face_id}"
            )

        success, encoded = cv2.imencode(
            ".jpg",
            crop,
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        )

        if not success:
            raise ValueError(
                f"Failed to encode thumbnail for face {face_id}"
            )

        return encoded.tobytes()