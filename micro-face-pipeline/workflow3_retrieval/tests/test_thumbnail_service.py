from __future__ import annotations

from pathlib import Path

import cv2

from app.db.postgres import SessionLocal
from app.repositories.dashboard_repository import DashboardRepository
from app.services.thumbnail_service import ThumbnailService


def test_thumbnail_service_face_105():
    db = SessionLocal()

    try:
        repository = DashboardRepository(db)
        service = ThumbnailService(repository)

        thumbnail = service.get_face_thumbnail(105)

        assert thumbnail is not None
        assert isinstance(thumbnail, bytes)
        assert len(thumbnail) > 0

        # Verify that the returned bytes are a valid JPEG image.
        image_array = __import__("numpy").frombuffer(
            thumbnail,
            dtype=__import__("numpy").uint8,
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR,
        )

        assert image is not None
        assert image.size > 0

        height, width = image.shape[:2]

        assert width > 0
        assert height > 0

        print(
            f"\nFace 105 thumbnail: "
            f"{width}x{height}, {len(thumbnail)} bytes"
        )

    finally:
        db.close()


def test_thumbnail_service_invalid_face():
    db = SessionLocal()

    try:
        repository = DashboardRepository(db)
        service = ThumbnailService(repository)

        thumbnail = service.get_face_thumbnail(999999)

        assert thumbnail is None

    finally:
        db.close()


def test_thumbnail_source_path_exists():
    db = SessionLocal()

    try:
        repository = DashboardRepository(db)

        data = repository.get_face_thumbnail_data(105)

        assert data is not None

        image_path = Path(data["local_path"])

        assert image_path.exists()
        assert image_path.is_file()

    finally:
        db.close()