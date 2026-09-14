from app.db.postgres import SessionLocal
from app.db.clustering_repository import ClusteringRepository


def main():

    print("=" * 70)
    print("WORKFLOW 2 POSTGRESQL REPOSITORY TEST")
    print("=" * 70)

    db = SessionLocal()

    try:

        repository = ClusteringRepository(db)

        print()
        print("[1] Reading face embeddings...")

        embeddings = (
            repository.get_all_face_embeddings()
        )

        print(
            f"Embedding metadata count: "
            f"{len(embeddings)}"
        )

        if len(embeddings) != 20:
            raise AssertionError(
                f"Expected 20 embeddings, "
                f"got {len(embeddings)}"
            )

        print(
            "Embedding count check: PASS"
        )

        print()
        print("[2] Checking first embedding...")

        first = embeddings[0]

        print(
            {
                "face_id": first.face_id,
                "vector_id": first.vector_id,
                "model_name": first.model_name,
                "model_version": first.model_version,
                "dimension": first.embedding_dimension,
                "vector_store": first.vector_store,
                "normalized": first.normalized,
                "norm": first.embedding_norm,
            }
        )

        assert first.embedding_dimension == 512
        assert first.vector_store == "hnsw"
        assert first.vector_id.startswith(
            "face:"
        )

        print(
            "Embedding metadata check: PASS"
        )

        print()
        print("[3] Checking clustering counts...")

        clusters = (
            repository.count_clusters()
        )

        memberships = (
            repository.count_memberships()
        )

        print(
            f"Clusters: {clusters}"
        )

        print(
            f"Memberships: {memberships}"
        )

        assert clusters == 0
        assert memberships == 0

        print(
            "Initial clustering state: PASS"
        )

        print()
        print("=" * 70)
        print(
            "WORKFLOW 2 POSTGRESQL REPOSITORY: PASS"
        )
        print("=" * 70)

    finally:

        db.close()


if __name__ == "__main__":
    main()