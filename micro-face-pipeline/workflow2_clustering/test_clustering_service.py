from __future__ import annotations

from pathlib import Path

import numpy as np

from app.clustering.clustering_service import (
    ClusteringService,
)
from app.clustering.hnsw_store import (
    HNSWVectorStore,
)
from app.clustering.vector_bootstrap import (
    Workflow1VectorReader,
)
from app.db.clustering_repository import (
    ClusteringRepository,
)
from app.db.postgres import SessionLocal


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent


WORKFLOW1_ROOT = (
    PROJECT_ROOT.parent
    / "workflow1_worker"
)


WORKFLOW1_INDEX = (
    WORKFLOW1_ROOT
    / "data"
    / "hnsw"
    / "index.bin"
)


WORKFLOW1_METADATA = (
    WORKFLOW1_ROOT
    / "data"
    / "hnsw"
    / "metadata.json"
)


TEST_HNSW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "hnsw"
    / "clustering_test"
)


TEST_INDEX = (
    TEST_HNSW_ROOT
    / "index.bin"
)


TEST_METADATA = (
    TEST_HNSW_ROOT
    / "metadata.json"
)


# ============================================================================
# TEST CLEANUP
# ============================================================================

def clean_test_hnsw() -> None:
    """
    Clean only the Workflow 2 test HNSW files.

    We deliberately do NOT remove the directory itself.

    On Windows, deleting the entire directory with shutil.rmtree()
    can fail with WinError 5 if the HNSW index was recently opened
    or is still referenced by a Python process.

    Removing only the files avoids unnecessary directory-level
    locking problems.
    """

    TEST_HNSW_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in (
        TEST_INDEX,
        TEST_METADATA,
    ):

        if path.exists():

            try:

                path.unlink()

            except PermissionError as exc:

                raise RuntimeError(
                    "\n"
                    "Workflow 2 HNSW test file is locked.\n"
                    "\n"
                    f"Locked file: {path}\n"
                    "\n"
                    "Close any Python process currently using "
                    "the Workflow 2 HNSW test index and run "
                    "the test again.\n"
                ) from exc


# ============================================================================
# MAIN TEST
# ============================================================================

def main():

    print("=" * 70)
    print(
        "WORKFLOW 2 REAL CLUSTERING TEST"
    )
    print("=" * 70)

    db = SessionLocal()

    repository = None
    active_store = None

    try:

        # ====================================================================
        # CREATE POSTGRESQL REPOSITORY
        # ====================================================================

        repository = ClusteringRepository(
            db
        )

        # ====================================================================
        # [0] RESET WORKFLOW 2 TEST STATE
        # ====================================================================

        print()
        print(
            "[0] Resetting Workflow 2 clustering state..."
        )

        repository.reset_clustering_state()

        print(
            "Workflow 2 PostgreSQL state reset: PASS"
        )

        # ====================================================================
        # [1] INITIAL POSTGRESQL STATE
        # ====================================================================

        print()
        print(
            "[1] Checking initial PostgreSQL state..."
        )

        embedding_records = (
            repository.get_all_face_embeddings()
        )

        embedding_count = len(
            embedding_records
        )

        cluster_count = (
            repository.count_clusters()
        )

        membership_count = (
            repository.count_memberships()
        )

        print(
            f"Embeddings:  {embedding_count}"
        )

        print(
            f"Clusters:    {cluster_count}"
        )

        print(
            f"Memberships: {membership_count}"
        )

        if embedding_count != 20:

            raise AssertionError(
                "Expected exactly 20 PostgreSQL "
                f"embedding records, found {embedding_count}"
            )

        if cluster_count != 0:

            raise AssertionError(
                "PostgreSQL must start with zero clusters. "
                f"Found {cluster_count}"
            )

        if membership_count != 0:

            raise AssertionError(
                "PostgreSQL must start with zero memberships. "
                f"Found {membership_count}"
            )

        print(
            "Initial PostgreSQL state: PASS"
        )

        # --------------------------------------------------------------------
        # This test intentionally expects the clean clustering state.
        # --------------------------------------------------------------------

        if embedding_count != 20:

            raise AssertionError(
                "Expected exactly 20 PostgreSQL "
                f"embedding records, found {embedding_count}"
            )

        if cluster_count != 0:

            raise AssertionError(
                "PostgreSQL must start with zero clusters. "
                f"Found {cluster_count}"
            )

        if membership_count != 0:

            raise AssertionError(
                "PostgreSQL must start with zero memberships. "
                f"Found {membership_count}"
            )

        print(
            "Initial PostgreSQL state: PASS"
        )

        # ====================================================================
        # [2] OPEN WORKFLOW 1 HNSW
        # ====================================================================

        print()
        print(
            "[2] Opening Workflow 1 HNSW..."
        )

        # --------------------------------------------------------------------
        # Workflow1VectorReader reads the actual embeddings produced by
        # Workflow 1.
        #
        # PostgreSQL supplies the authoritative embedding metadata.
        # Workflow 1 HNSW supplies the actual 512-D vectors.
        # --------------------------------------------------------------------

        source_store = (
            Workflow1VectorReader(
                index_path=WORKFLOW1_INDEX,
                metadata_path=WORKFLOW1_METADATA,
                dimension=512,
            )
        )

        source_count = (
            source_store.count()
        )

        print(
            f"Workflow 1 vectors: {source_count}"
        )

        if source_count != 20:

            raise AssertionError(
                "Expected exactly 20 vectors in "
                f"Workflow 1 HNSW, found {source_count}"
            )

        print(
            "Workflow 1 source: PASS"
        )

        # ====================================================================
        # [3] CREATE CLEAN WORKFLOW 2 HNSW
        # ====================================================================

        print()
        print(
            "[3] Creating clean Workflow 2 HNSW..."
        )

        active_store = (
            HNSWVectorStore(
                index_path=TEST_INDEX,
                metadata_path=TEST_METADATA,
                dimension=512,
            )
        )

        active_count = (
            active_store.count()
        )

        print(
            f"Workflow 2 HNSW vectors: "
            f"{active_count}"
        )

        if active_count != 0:

            raise AssertionError(
                "Workflow 2 HNSW must start empty. "
                f"Found {active_count} vectors."
            )

        print(
            "Workflow 2 HNSW empty: PASS"
        )

        # ====================================================================
        # [4] CREATE CLUSTERING SERVICE
        # ====================================================================

        print()
        print(
            "[4] Creating ClusteringService..."
        )

        service = ClusteringService(
            repository=repository,
            source_store=source_store,
            active_store=active_store,
            similarity_threshold=0.50,
            search_k=5,
        )

        print(
            "ClusteringService: PASS"
        )

        # ====================================================================
        # [5] CLUSTER ALL REAL EMBEDDINGS
        # ====================================================================

        print()
        print(
            "[5] Clustering 20 real embeddings..."
        )

        results = (
            service.process_all()
        )

        # --------------------------------------------------------------------
        # We expect exactly one result per PostgreSQL embedding.
        # --------------------------------------------------------------------

        if len(results) != 20:

            raise AssertionError(
                "Expected 20 clustering results, "
                f"received {len(results)}"
            )

        print()

        for result in results:

            print(
                f"face_id={result.face_id:>3} "
                f"vector_id={result.vector_id:<10} "
                f"cluster_id={result.cluster_id:>3} "
                f"similarity={result.similarity:.6f} "
                f"new_cluster="
                f"{result.created_new_cluster}"
            )

        # ====================================================================
        # [6] FINAL COUNTS
        # ====================================================================

        print()
        print(
            "[6] Checking final counts..."
        )

        final_clusters = (
            repository.count_clusters()
        )

        final_memberships = (
            repository.count_memberships()
        )

        final_hnsw_count = (
            active_store.count()
        )

        print(
            f"Clusters:    {final_clusters}"
        )

        print(
            f"Memberships: {final_memberships}"
        )

        print(
            f"HNSW vectors: {final_hnsw_count}"
        )

        # --------------------------------------------------------------------
        # We must have:
        #
        #   1 <= clusters <= 20
        #   memberships = 20
        #   HNSW vectors = 20
        #
        # We intentionally do not require exactly 5 clusters here.
        # The clustering algorithm should determine the result.
        # --------------------------------------------------------------------

        if not (
            1
            <= final_clusters
            <= 20
        ):

            raise AssertionError(
                "Invalid cluster count: "
                f"{final_clusters}"
            )

        if final_memberships != 20:

            raise AssertionError(
                "Expected 20 cluster memberships, "
                f"found {final_memberships}"
            )

        if final_hnsw_count != 20:

            raise AssertionError(
                "Expected 20 Workflow 2 HNSW vectors, "
                f"found {final_hnsw_count}"
            )

        print(
            "Final count checks: PASS"
        )

        # ====================================================================
        # [7] VERIFY EVERY FACE HAS EXACTLY ONE CLUSTER
        # ====================================================================

        print()
        print(
            "[7] Verifying every face has one cluster..."
        )

        # --------------------------------------------------------------------
        # Store the authoritative PostgreSQL face -> cluster mapping.
        # --------------------------------------------------------------------

        seen_clusters: dict[
            int,
            int
        ] = {}

        for record in embedding_records:

            cluster_id = (
                repository
                .get_cluster_for_face(
                    record.face_id
                )
            )

            if cluster_id is None:

                raise AssertionError(
                    "Face has no cluster assignment: "
                    f"face_id={record.face_id}"
                )

            seen_clusters[
                record.face_id
            ] = int(
                cluster_id
            )

        if len(seen_clusters) != 20:

            raise AssertionError(
                "Expected 20 face-to-cluster assignments, "
                f"found {len(seen_clusters)}"
            )

        print(
            "All 20 faces assigned: PASS"
        )

        # ====================================================================
        # [8] VERIFY POSTGRESQL <-> HNSW CLUSTER MAPPINGS
        # ====================================================================

        print()
        print(
            "[8] Verifying HNSW cluster mappings..."
        )

        # --------------------------------------------------------------------
        # Every vector in Workflow 2 HNSW must carry the same cluster_id
        # as PostgreSQL.
        # --------------------------------------------------------------------

        for record in embedding_records:

            hnsw_cluster = (
                active_store
                .vector_to_cluster
                .get(
                    record.vector_id
                )
            )

            db_cluster = (
                seen_clusters[
                    record.face_id
                ]
            )

            if hnsw_cluster is None:

                raise AssertionError(
                    "Vector has no HNSW cluster mapping: "
                    f"vector_id={record.vector_id}"
                )

            if int(hnsw_cluster) != int(
                db_cluster
            ):

                raise AssertionError(
                    "PostgreSQL/HNSW cluster mismatch: "
                    f"face_id={record.face_id}, "
                    f"vector_id={record.vector_id}, "
                    f"postgres={db_cluster}, "
                    f"hnsw={hnsw_cluster}"
                )

        print(
            "PostgreSQL/HNSW cluster mappings: PASS"
        )

        # ====================================================================
        # [9] VERIFY VECTOR SEARCH + CLUSTER CONSISTENCY
        # ====================================================================

        print()
        print(
            "[9] Verifying vector identity/search..."
        )

        # --------------------------------------------------------------------
        # IMPORTANT:
        #
        # The nearest vector does NOT necessarily have to be the exact
        # queried vector_id.
        #
        # Multiple faces can have identical embeddings.
        #
        # Example from the current test data:
        #
        #   face:6 and face:1 have similarity = 1.0
        #
        # Therefore, when searching for face:6, HNSW may legitimately
        # return face:1.
        #
        # The correct invariant is:
        #
        #   1. nearest similarity is approximately 1.0
        #   2. nearest vector belongs to the same cluster
        #
        # This validates identity consistency rather than relying on
        # vector_id uniqueness among identical vectors.
        # --------------------------------------------------------------------

        for record in embedding_records:

            vector = (
                source_store
                .get_embedding(
                    record.vector_id
                )
            )

            if vector is None:

                raise AssertionError(
                    "Workflow 1 embedding missing: "
                    f"vector_id={record.vector_id}"
                )

            vector = np.asarray(
                vector,
                dtype=np.float32,
            ).reshape(-1)

            # ---------------------------------------------------------------
            # Validate vector shape.
            # ---------------------------------------------------------------

            if vector.shape != (
                512,
            ):

                raise AssertionError(
                    "Unexpected vector shape: "
                    f"vector_id={record.vector_id}, "
                    f"shape={vector.shape}"
                )

            # ---------------------------------------------------------------
            # Search Workflow 2.
            # ---------------------------------------------------------------

            search_results = (
                active_store.search(
                    vector,
                    k=1,
                )
            )

            if not search_results:

                raise AssertionError(
                    "No HNSW search result returned: "
                    f"vector_id={record.vector_id}"
                )

            nearest = (
                search_results[0]
            )

            nearest_vector_id = (
                nearest["vector_id"]
            )

            nearest_similarity = float(
                nearest["similarity"]
            )

            nearest_cluster = int(
                nearest["cluster_id"]
            )

            expected_cluster = (
                seen_clusters[
                    record.face_id
                ]
            )

            print(
                f"face_id={record.face_id:>3} "
                f"query={record.vector_id:<10} "
                f"nearest={nearest_vector_id:<10} "
                f"similarity={nearest_similarity:.6f} "
                f"expected_cluster="
                f"{expected_cluster} "
                f"nearest_cluster="
                f"{nearest_cluster}"
            )

            # ---------------------------------------------------------------
            # Identical embeddings should retrieve ~1.0 similarity.
            # ---------------------------------------------------------------

            if not np.isclose(
                nearest_similarity,
                1.0,
                atol=1e-4,
            ):

                raise AssertionError(
                    "Nearest similarity is not approximately 1.0: "
                    f"query={record.vector_id}, "
                    f"nearest={nearest_vector_id}, "
                    f"similarity={nearest_similarity}"
                )

            # ---------------------------------------------------------------
            # Most important check:
            #
            # nearest vector must belong to the same identity cluster.
            # ---------------------------------------------------------------

            if (
                nearest_cluster
                != expected_cluster
            ):

                raise AssertionError(
                    "Nearest vector belongs to the wrong cluster: "
                    f"query={record.vector_id}, "
                    f"nearest={nearest_vector_id}, "
                    f"expected_cluster={expected_cluster}, "
                    f"actual_cluster={nearest_cluster}"
                )

        print(
            "All 20 vectors searchable and "
            "cluster-consistent: PASS"
        )

        # ====================================================================
        # [10] IDEMPOTENCY TEST
        # ====================================================================

        print()
        print(
            "[10] Testing clustering idempotency..."
        )

        # --------------------------------------------------------------------
        # Run clustering a second time against the same PostgreSQL records.
        #
        # Existing face memberships should be detected by the repository,
        # and the service must NOT create duplicate memberships or vectors.
        # --------------------------------------------------------------------

        second_run = (
            service.process_all()
        )

        if len(second_run) != 20:

            raise AssertionError(
                "Expected 20 results during second run, "
                f"received {len(second_run)}"
            )

        clusters_after_second_run = (
            repository.count_clusters()
        )

        memberships_after_second_run = (
            repository.count_memberships()
        )

        hnsw_after_second_run = (
            active_store.count()
        )

        print(
            f"Clusters after rerun: "
            f"{clusters_after_second_run}"
        )

        print(
            f"Memberships after rerun: "
            f"{memberships_after_second_run}"
        )

        print(
            f"HNSW vectors after rerun: "
            f"{hnsw_after_second_run}"
        )

        # --------------------------------------------------------------------
        # No new clusters.
        # --------------------------------------------------------------------

        if (
            clusters_after_second_run
            != final_clusters
        ):

            raise AssertionError(
                "Idempotency failure: cluster count changed. "
                f"before={final_clusters}, "
                f"after={clusters_after_second_run}"
            )

        # --------------------------------------------------------------------
        # No duplicate memberships.
        # --------------------------------------------------------------------

        if (
            memberships_after_second_run
            != 20
        ):

            raise AssertionError(
                "Idempotency failure: membership count changed. "
                f"Expected=20, "
                f"actual={memberships_after_second_run}"
            )

        # --------------------------------------------------------------------
        # No duplicate HNSW vectors.
        # --------------------------------------------------------------------

        if (
            hnsw_after_second_run
            != 20
        ):

            raise AssertionError(
                "Idempotency failure: HNSW vector count changed. "
                f"Expected=20, "
                f"actual={hnsw_after_second_run}"
            )

        print(
            "Idempotency: PASS"
        )

        # ====================================================================
        # FINAL SUCCESS
        # ====================================================================

        print()
        print("=" * 70)
        print(
            "WORKFLOW 2 REAL CLUSTERING: PASS"
        )
        print("=" * 70)

    except Exception:

        if repository is not None:

            repository.rollback()

        raise

    finally:

        # --------------------------------------------------------------------
        # Explicitly release the HNSW store reference.
        #
        # This is particularly useful on Windows because the underlying
        # index files can otherwise remain locked while the object is alive.
        # --------------------------------------------------------------------

        if active_store is not None:

            try:

                active_store.save()

            except Exception:
                pass

            del active_store

        db.close()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()