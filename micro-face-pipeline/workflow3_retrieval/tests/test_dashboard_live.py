from __future__ import annotations

from pathlib import Path

from app.db.postgres import SessionLocal
from app.repositories.dashboard_repository import DashboardRepository
from app.services.dashboard_service import DashboardService


def test_dashboard_summary_is_valid():
    db = SessionLocal()
    try:
        service = DashboardService(DashboardRepository(db))
        summary = service.get_summary()

        assert summary["total_faces"] >= 0
        assert summary["total_clusters"] >= 0
        assert summary["total_images"] >= 0

        assert summary["total_faces"] >= summary["total_clusters"]

        for item in summary["gender_distribution"].values():
            assert item >= 0

        for item in summary["age_group_distribution"].values():
            assert item >= 0
    finally:
        db.close()


def test_cluster_list_matches_summary():
    db = SessionLocal()
    try:
        service = DashboardService(DashboardRepository(db))

        summary = service.get_summary()
        clusters = service.get_clusters()

        assert len(clusters) == summary["total_clusters"]

        cluster_ids = [c["cluster_id"] for c in clusters]

        assert len(cluster_ids) == len(set(cluster_ids))

        for cluster in clusters:
            assert cluster["cluster_id"] > 0
            assert cluster["face_count"] >= 0
    finally:
        db.close()


def test_cluster_details_are_consistent():
    db = SessionLocal()
    try:
        service = DashboardService(DashboardRepository(db))

        clusters = service.get_clusters()

        for cluster in clusters:
            cluster_id = cluster["cluster_id"]
            detail = service.get_cluster_detail(cluster_id)

            assert detail is not None
            assert detail["cluster_id"] == cluster_id
            assert detail["face_count"] == len(detail["faces"])

            face_ids = [face["face_id"] for face in detail["faces"]]
            assert len(face_ids) == len(set(face_ids))

            for face in detail["faces"]:
                assert face["face_id"] > 0
                assert face["image_id"] > 0
                assert face["face_index"] >= 0
                assert face["similarity"] >= 0.0
    finally:
        db.close()


def test_image_faces_are_consistent():
    from sqlalchemy import text

    db = SessionLocal()
    try:
        repository = DashboardRepository(db)
        service = DashboardService(repository)

        rows = db.execute(
            text("""
                SELECT id
                FROM face_pipeline.images
                ORDER BY id
            """)
        ).scalars().all()

        for image_id in rows:
            result = service.get_image_faces(image_id)

            assert result is not None
            assert result["image_id"] == image_id
            assert result["total_faces"] == len(result["faces"])

            indices = [
                face["face_index"]
                for face in result["faces"]
            ]

            assert len(indices) == len(set(indices))

            for face in result["faces"]:
                assert face["image_id"] == image_id
                assert face["bbox_x2"] > face["bbox_x1"]
                assert face["bbox_y2"] > face["bbox_y1"]
    finally:
        db.close()


def test_face_105_thumbnail_metadata():
    db = SessionLocal()
    try:
        repository = DashboardRepository(db)

        data = repository.get_face_thumbnail_data(105)

        assert data is not None
        assert data["face_id"] == 105
        assert data["image_id"] > 0

        path = Path(data["local_path"])

        assert path.exists()

        assert data["bbox_x2"] > data["bbox_x1"]
        assert data["bbox_y2"] > data["bbox_y1"]
    finally:
        db.close()