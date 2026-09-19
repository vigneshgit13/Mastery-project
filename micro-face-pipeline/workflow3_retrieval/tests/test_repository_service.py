from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

from app.db.postgres import SessionLocal
from app.repositories.dashboard_repository import DashboardRepository
from app.services.dashboard_service import DashboardService


def main() -> None:
    print("=" * 70)
    print("WORKFLOW 3 REPOSITORY + SERVICE REGRESSION")
    print("=" * 70)

    db = SessionLocal()

    try:
        repository = DashboardRepository(db)
        service = DashboardService(repository)

        # --------------------------------------------------------------
        # 1. Dashboard summary
        # --------------------------------------------------------------
        print("\n[1] Dashboard summary")

        summary = service.get_summary()

        assert summary["total_faces"] == 194
        assert summary["total_clusters"] == 100
        assert summary["total_images"] == 14

        print("PASS  Summary counts")
        print(f"      Faces:    {summary['total_faces']}")
        print(f"      Clusters: {summary['total_clusters']}")
        print(f"      Images:   {summary['total_images']}")

        # --------------------------------------------------------------
        # 2. Cluster list
        # --------------------------------------------------------------
        print("\n[2] Cluster list")

        clusters = service.get_clusters()

        assert len(clusters) == 100

        print(f"PASS  Cluster count: {len(clusters)}")

        # --------------------------------------------------------------
        # 3. Cluster 1
        # --------------------------------------------------------------
        print("\n[3] Cluster 1 detail")

        cluster = service.get_cluster_detail(1)

        assert cluster is not None
        assert cluster["cluster_id"] == 1
        assert cluster["face_count"] == len(cluster["faces"])

        print(
            f"PASS  Cluster 1: "
            f"{cluster['face_count']} faces"
        )

        # --------------------------------------------------------------
        # 4. Cluster 1 images
        # --------------------------------------------------------------
        print("\n[4] Cluster 1 images")

        images = service.get_cluster_images(1)

        image_ids = [image["image_id"] for image in images]

        assert len(image_ids) == len(set(image_ids))

        print(
            f"PASS  Cluster 1 images: "
            f"{len(images)}"
        )

        # --------------------------------------------------------------
        # 5. Image 1 faces
        # --------------------------------------------------------------
        print("\n[5] Image 1 faces")

        image_faces = service.get_image_faces(1)

        assert image_faces is not None
        assert image_faces["image_id"] == 1
        assert image_faces["total_faces"] == 10
        assert len(image_faces["faces"]) == 10

        face_indices = [
            face["face_index"]
            for face in image_faces["faces"]
        ]

        assert face_indices == list(range(10))
        assert len(face_indices) == len(set(face_indices))

        print("PASS  Image 1: 10 faces")
        print("PASS  Face indices: 0-9")
        print("PASS  No duplicate face indices")

        # --------------------------------------------------------------
        # 6. Face 105 thumbnail metadata
        # --------------------------------------------------------------
        print("\n[6] Face 105 thumbnail metadata")

        thumbnail_data = repository.get_face_thumbnail_data(105)

        assert thumbnail_data is not None
        assert thumbnail_data["face_id"] == 105
        assert thumbnail_data["image_id"] == 7
        assert thumbnail_data["local_path"]

        assert thumbnail_data["bbox_x2"] > thumbnail_data["bbox_x1"]
        assert thumbnail_data["bbox_y2"] > thumbnail_data["bbox_y1"]

        print("PASS  Face 105 exists")
        print("PASS  Image relationship: 105 -> 7")
        print("PASS  Bounding box is valid")
        print("PASS  Local image path exists in metadata")

        # --------------------------------------------------------------
        # 7. Invalid face
        # --------------------------------------------------------------
        print("\n[7] Invalid face")

        invalid = repository.get_face_thumbnail_data(999999)

        assert invalid is None

        print("PASS  Invalid face returns None")

        # --------------------------------------------------------------
        # Final
        # --------------------------------------------------------------
        print("\n" + "=" * 70)
        print("WORKFLOW 3 REPOSITORY + SERVICE REGRESSION: PASS")
        print("=" * 70)

    finally:
        db.close()


if __name__ == "__main__":
    main()