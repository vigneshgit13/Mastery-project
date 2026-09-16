from app.db.postgres import SessionLocal
from app.repositories.dashboard_repository import DashboardRepository


print("=" * 70)
print("WORKFLOW 3 CLUSTER IMAGES REPOSITORY TEST")
print("=" * 70)

db = SessionLocal()

try:
    repository = DashboardRepository(db)

    images = repository.get_cluster_images(1)

    print()
    print(f"Images returned: {len(images)}")

    assert len(images) > 0

    image_ids = [image["image_id"] for image in images]

    # DISTINCT image retrieval must not return duplicates.
    assert len(image_ids) == len(set(image_ids))

    print()
    print("Images:")

    for image in images:
        print(
            f"  Image {image['image_id']} | "
            f"Upload {image['upload_id']} | "
            f"{image['width']}x{image['height']} | "
            f"{image['content_type']} | "
            f"{image['bucket']}/{image['blob_name']}"
        )

        assert image["image_id"] is not None
        assert image["upload_id"] is not None
        assert image["bucket"] is not None
        assert image["blob_name"] is not None

    print()
    print("Validation:")
    print("  Images retrieved:    PASS")
    print("  No duplicate images: PASS")
    print("  Image references:    PASS")
    print("  Source metadata:     PASS")

    print()
    print("WORKFLOW 3 CLUSTER IMAGES REPOSITORY: PASS")

finally:
    db.close()