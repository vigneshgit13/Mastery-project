from app.db.postgres import SessionLocal
from app.repositories.dashboard_repository import DashboardRepository


print("=" * 70)
print("WORKFLOW 3 CLUSTER REPOSITORY TEST")
print("=" * 70)

db = SessionLocal()

try:
    repository = DashboardRepository(db)

    clusters = repository.get_clusters()

    print()
    print(f"Total clusters returned: {len(clusters)}")

    assert len(clusters) > 0

    print()
    print("First 5 clusters:")

    for cluster in clusters[:5]:
        print(
            f"  Cluster {cluster['cluster_id']}: "
            f"{cluster['face_count']} faces"
        )

    total_faces_in_clusters = sum(
        cluster["face_count"]
        for cluster in clusters
    )

    print()
    print(
        f"Faces across all clusters: "
        f"{total_faces_in_clusters}"
    )

    assert total_faces_in_clusters > 0

    print()
    print("Validation:")
    print("  Cluster count:       PASS")
    print("  Cluster face counts: PASS")
    print("  Total memberships:   PASS")

    print()
    print("WORKFLOW 3 CLUSTER REPOSITORY: PASS")

finally:
    db.close()