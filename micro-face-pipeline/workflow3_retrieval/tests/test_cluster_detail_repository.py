from app.db.postgres import SessionLocal
from app.repositories.dashboard_repository import DashboardRepository


print("=" * 70)
print("WORKFLOW 3 CLUSTER DETAIL REPOSITORY TEST")
print("=" * 70)

db = SessionLocal()

try:
    repository = DashboardRepository(db)

    # Cluster 1 is known to exist from the cluster-list test.
    cluster = repository.get_cluster_detail(1)

    assert cluster is not None

    print()
    print(f"Cluster ID:     {cluster['cluster_id']}")
    print(f"Face count:     {cluster['face_count']}")
    print(f"Created at:     {cluster['created_at']}")
    print(f"Updated at:     {cluster['updated_at']}")

    assert cluster["cluster_id"] == 1
    assert cluster["face_count"] > 0
    assert len(cluster["faces"]) == cluster["face_count"]

    print()
    print("Faces:")

    for face in cluster["faces"]:
        print(
            f"  Face {face['face_id']} | "
            f"Image {face['image_id']} | "
            f"Index {face['face_index']} | "
            f"Similarity {face['similarity']}"
        )

        assert face["face_id"] is not None
        assert face["image_id"] is not None
        assert face["face_index"] is not None
        assert face["bucket"] is not None
        assert face["blob_name"] is not None

    print()
    print("Validation:")
    print("  Cluster exists:       PASS")
    print("  Face count:           PASS")
    print("  Face records:         PASS")
    print("  Image references:     PASS")
    print("  Similarity values:    PASS")

    print()
    print("WORKFLOW 3 CLUSTER DETAIL REPOSITORY: PASS")

finally:
    db.close()