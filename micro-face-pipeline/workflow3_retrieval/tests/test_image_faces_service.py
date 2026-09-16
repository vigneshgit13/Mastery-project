from app.db.postgres import SessionLocal
from app.repositories.dashboard_repository import DashboardRepository
from app.services.dashboard_service import DashboardService


def main() -> None:
    print("=" * 70)
    print("WORKFLOW 3 IMAGE FACES SERVICE TEST")
    print("=" * 70)

    db = SessionLocal()

    try:
        repository = DashboardRepository(db)
        service = DashboardService(repository)

        image_id = 1

        result = service.get_image_faces(image_id)

        assert result is not None
        assert result["image_id"] == image_id
        assert result["total_faces"] == len(result["faces"])
        assert result["total_faces"] > 0

        indices = [face["face_index"] for face in result["faces"]]

        assert indices == sorted(indices)
        assert len(indices) == len(set(indices))

        for face in result["faces"]:
            assert face["image_id"] == image_id

            assert face["bbox_x1"] <= face["bbox_x2"]
            assert face["bbox_y1"] <= face["bbox_y2"]

            assert face["detection_confidence"] is not None

        print()
        print(f"Image ID:             {result['image_id']}")
        print(f"Faces returned:       {result['total_faces']}")

        print()
        print("Validation:")
        print("  Image reference:       PASS")
        print("  Face count:            PASS")
        print("  Face ordering:         PASS")
        print("  No duplicate indices:  PASS")
        print("  Bounding boxes:        PASS")
        print("  Detection confidence:  PASS")

        print()
        print("WORKFLOW 3 IMAGE FACES SERVICE: PASS")

    finally:
        db.close()


if __name__ == "__main__":
    main()