from app.db.postgres import SessionLocal
from app.repositories.dashboard_repository import DashboardRepository
from app.services.dashboard_service import DashboardService


print("=" * 70)
print("WORKFLOW 3 CLUSTER DETAIL SERVICE TEST")
print("=" * 70)

db = SessionLocal()

try:
    repository = DashboardRepository(db)
    service = DashboardService(repository)

    cluster = service.get_cluster_detail(1)

    assert cluster is not None

    print()
    print(f"Cluster ID:     {cluster['cluster_id']}")
    print(f"Face count:     {cluster['face_count']}")

    assert cluster["cluster_id"] == 1
    assert cluster["face_count"] == len(cluster["faces"])
    assert cluster["face_count"] > 0

    print()
    print("Faces returned:")

    for face in cluster["faces"]:
        print(
            f"  Face {face['face_id']} | "
            f"Image {face['image_id']} | "
            f"Similarity {face['similarity']}"
        )

    print()
    print("Validation:")
    print("  Service retrieval: PASS")
    print("  Cluster identity:  PASS")
    print("  Face records:      PASS")

    print()
    print("WORKFLOW 3 CLUSTER DETAIL SERVICE: PASS")

finally:
    db.close()