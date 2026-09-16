from app.db.postgres import SessionLocal
from app.repositories.dashboard_repository import DashboardRepository


print("=" * 70)
print("WORKFLOW 3 IMAGE FACES REPOSITORY TEST")
print("=" * 70)

db = SessionLocal()

try:
    repository = DashboardRepository(db)

    faces = repository.get_image_faces(1)

    print()
    print(f"Faces returned: {len(faces)}")

    assert len(faces) == 10

    face_indices = [face["face_index"] for face in faces]

    assert face_indices == sorted(face_indices)
    assert len(face_indices) == len(set(face_indices))

    print()
    print("Faces:")

    for face in faces:
        print(
            f"  Face {face['face_id']} | "
            f"Index {face['face_index']} | "
            f"Gender {face['gender']} | "
            f"Age {face['age_years']} | "
            f"Group {face['age_group']} | "
            f"Detection {face['detection_confidence']}"
        )

        assert face["face_id"] is not None
        assert face["image_id"] == 1
        assert face["face_index"] is not None

        assert face["bbox_x1"] is not None
        assert face["bbox_y1"] is not None
        assert face["bbox_x2"] is not None
        assert face["bbox_y2"] is not None

        assert face["detection_confidence"] is not None

    print()
    print("Validation:")
    print("  Faces retrieved:       PASS")
    print("  Image references:      PASS")
    print("  Face ordering:         PASS")
    print("  No duplicate indices:  PASS")
    print("  Bounding boxes:        PASS")
    print("  Detection confidence:  PASS")

    print()
    print("WORKFLOW 3 IMAGE FACES REPOSITORY: PASS")

finally:
    db.close()