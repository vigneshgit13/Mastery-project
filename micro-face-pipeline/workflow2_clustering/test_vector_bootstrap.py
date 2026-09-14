from __future__ import annotations
import numpy as np
from pathlib import Path

from app.clustering.hnsw_store import HNSWVectorStore
from app.clustering.vector_bootstrap import (
    Workflow1ToWorkflow2Bootstrap,
    Workflow1VectorReader,
)
from app.db.clustering_repository import (
    ClusteringRepository,
)
from app.db.postgres import SessionLocal


PROJECT_ROOT = Path(__file__).resolve().parent

WORKFLOW1_ROOT = (
    PROJECT_ROOT.parent / "workflow1_worker"
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

WORKFLOW2_INDEX = (
    PROJECT_ROOT
    / "data"
    / "hnsw"
    / "bootstrap_test"
    / "index.bin"
)

WORKFLOW2_METADATA = (
    PROJECT_ROOT
    / "data"
    / "hnsw"
    / "bootstrap_test"
    / "metadata.json"
)


def main():

    print("=" * 70)
    print(
        "WORKFLOW 1 -> WORKFLOW 2 "
        "REAL VECTOR BOOTSTRAP TEST"
    )
    print("=" * 70)

    # --------------------------------------------------------------
    # Paths
    # --------------------------------------------------------------

    print()
    print("[1] Checking Workflow 1 vector store...")

    print(
        f"Index:    {WORKFLOW1_INDEX}"
    )

    print(
        f"Metadata: {WORKFLOW1_METADATA}"
    )

    assert WORKFLOW1_INDEX.exists()
    assert WORKFLOW1_METADATA.exists()

    print(
        "Workflow 1 files: PASS"
    )

    # --------------------------------------------------------------
    # Source
    # --------------------------------------------------------------

    print()
    print(
        "[2] Opening Workflow 1 HNSW..."
    )

    source = Workflow1VectorReader(
        index_path=WORKFLOW1_INDEX,
        metadata_path=WORKFLOW1_METADATA,
        dimension=512,
    )

    print(
        f"Workflow 1 index vectors: "
        f"{source.index.ntotal}"
    )

    print(
        f"Workflow 1 metadata vectors: "
        f"{len(source.vector_to_faiss)}"
    )

    assert source.index.ntotal == 20
    assert len(source.vector_to_faiss) == 20

    print(
        "Workflow 1 vector count: PASS"
    )

    # --------------------------------------------------------------
    # Database
    # --------------------------------------------------------------

    print()
    print(
        "[3] Reading PostgreSQL embedding metadata..."
    )

    db = SessionLocal()

    try:

        repository = ClusteringRepository(
            db
        )

        records = (
            repository.get_all_face_embeddings()
        )

        print(
            f"PostgreSQL embedding records: "
            f"{len(records)}"
        )

        assert len(records) == 20

        print(
            "PostgreSQL count: PASS"
        )

        # ----------------------------------------------------------
        # Target
        # ----------------------------------------------------------

        print()
        print(
            "[4] Creating clean Workflow 2 HNSW..."
        )

        if WORKFLOW2_INDEX.exists():
            WORKFLOW2_INDEX.unlink()

        if WORKFLOW2_METADATA.exists():
            WORKFLOW2_METADATA.unlink()

        WORKFLOW2_INDEX.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        target = HNSWVectorStore(
            index_path=WORKFLOW2_INDEX,
            metadata_path=WORKFLOW2_METADATA,
            dimension=512,
        )

        assert target.count() == 0

        print(
            "Clean target HNSW: PASS"
        )

        # ----------------------------------------------------------
        # Bootstrap
        # ----------------------------------------------------------

        print()
        print(
            "[5] Bootstrapping real vectors..."
        )

        bootstrap = (
            Workflow1ToWorkflow2Bootstrap(
                repository=repository,
                source_store=source,
                target_store=target,
            )
        )

        result = bootstrap.run()

        print(
            f"Bootstrap result: {result}"
        )

        assert result[
            "source_records"
        ] == 20

        assert result[
            "added"
        ] == 20

        assert result[
            "skipped_existing"
        ] == 0

        assert result[
            "target_vectors"
        ] == 20

        print(
            "Bootstrap count: PASS"
        )

        # ----------------------------------------------------------
        # Verify every vector.
        # ----------------------------------------------------------

        print()
        print(
            "[6] Verifying every real vector..."
        )

        for record in records:

            vector = target.get_embedding(
                record.vector_id
            )

            assert vector is not None

            assert vector.shape == (
                512,
            )

            print(
                f"face_id={record.face_id:>3} "
                f"vector_id={record.vector_id:<10} "
                f"norm={float((vector ** 2).sum() ** 0.5):.6f}"
            )

        print(
            "All 20 vector retrievals: PASS"
        )

        # ----------------------------------------------------------
        # Verify vector identity between source and target.
        # ----------------------------------------------------------

        print()
        print(
            "[7] Comparing source and target vectors..."
        )

        for record in records:

            source_vector = (
                source.get_embedding(
                    record.vector_id
                )
            )

            target_vector = (
                target.get_embedding(
                    record.vector_id
                )
            )

            assert source_vector is not None
            assert target_vector is not None

            

            similarity = float(
                np.dot(
                    source_vector,
                    target_vector,
                )
            )

            if not np.isclose(
                similarity,
                1.0,
                atol=1e-5,
            ):
                raise AssertionError(
                    "Source/target vector mismatch: "
                    f"{record.vector_id}, "
                    f"similarity={similarity}"
                )

        print(
            "Source/target vector identity: PASS"
        )

        # ----------------------------------------------------------
        # Verify search.
        # ----------------------------------------------------------

        print()
        print(
            "[8] Testing Workflow 2 nearest-neighbor search..."
        )

        first = records[0]

        query = source.get_embedding(
            first.vector_id
        )

        assert query is not None

        results = target.search(
            query,
            k=3,
        )

        print(
            "Search results:"
        )

        for result_item in results:
            print(
                result_item
            )

        assert len(results) > 0

        assert (
            results[0]["vector_id"]
            == first.vector_id
        )

        assert np.isclose(
            results[0]["similarity"],
            1.0,
            atol=1e-4,
        )

        print(
            "Nearest-neighbor identity: PASS"
        )

        print()
        print("=" * 70)
        print(
            "REAL VECTOR BOOTSTRAP: PASS"
        )
        print("=" * 70)

    finally:

        db.close()


if __name__ == "__main__":
    main()
