from app.db.postgres import SessionLocal
from app.repositories.dashboard_repository import DashboardRepository
from app.services.dashboard_service import DashboardService


print("=" * 70)
print("WORKFLOW 3 CLUSTER SERVICE TEST")
print("=" * 70)

db = SessionLocal()

try:
    repository = DashboardRepository(db)
    service = DashboardService(repository)

    clusters = service.get_clusters()

    print()
    print(f"Total clusters returned: {len(clusters)}")

    assert len(clusters) == 86

    total_faces = sum(
        cluster["face_count"]
        for cluster in clusters
    )

    print(f"Faces across clusters:   {total_faces}")

    assert total_faces == 149

    print()
    print("Validation:")
    print("  Service retrieval:  PASS")
    print("  Cluster count:      PASS")
    print("  Face membership:    PASS")

    print()
    print("WORKFLOW 3 CLUSTER SERVICE: PASS")

finally:
    db.close()