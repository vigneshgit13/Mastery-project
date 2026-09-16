from app.db.postgres import SessionLocal
from app.repositories.dashboard_repository import DashboardRepository
from app.services.dashboard_service import DashboardService


print("=" * 70)
print("WORKFLOW 3 CLUSTER IMAGES SERVICE TEST")
print("=" * 70)

db = SessionLocal()

try:
    repository = DashboardRepository(db)
    service = DashboardService(repository)

    images = service.get_cluster_images(1)

    print()
    print(f"Images returned: {len(images)}")

    assert len(images) == 3

    image_ids = [image["image_id"] for image in images]

    assert image_ids == [1, 2, 3]
    assert len(image_ids) == len(set(image_ids))

    print()
    print("Images:")

    for image in images:
        print(
            f"  Image {image['image_id']} | "
            f"Upload {image['upload_id']} | "
            f"{image['width']}x{image['height']}"
        )

    print()
    print("Validation:")
    print("  Service retrieval:   PASS")
    print("  Image count:         PASS")
    print("  Image ordering:      PASS")
    print("  No duplicates:       PASS")

    print()
    print("WORKFLOW 3 CLUSTER IMAGES SERVICE: PASS")

finally:
    db.close()