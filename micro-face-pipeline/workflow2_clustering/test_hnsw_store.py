from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np

from app.clustering.hnsw_store import HNSWVectorStore


def main() -> None:
    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="workflow2_hnsw_test_"
        )
    )

    print("=" * 70)
    print("WORKFLOW 2 HNSW VECTOR STORE TEST")
    print("=" * 70)

    try:
        index_path = temp_dir / "index.bin"
        metadata_path = temp_dir / "metadata.json"

        print("\n[1] Creating HNSWVectorStore...")

        store = HNSWVectorStore(
            index_path=index_path,
            metadata_path=metadata_path,
            dimension=512,
        )

        print(
            "Initial vector count:",
            store.count(),
        )

        # ----------------------------------------------------------
        # Create two deterministic normalized embeddings.
        # ----------------------------------------------------------

        embedding_a = np.zeros(
            512,
            dtype=np.float32,
        )

        embedding_b = np.zeros(
            512,
            dtype=np.float32,
        )

        embedding_a[0] = 1.0
        embedding_b[1] = 1.0

        # ----------------------------------------------------------
        # ADD
        # ----------------------------------------------------------

        print("\n[2] Adding face:101...")

        added = store.add(
            embedding=embedding_a,
            vector_id="face:101",
            cluster_id=1,
        )

        print(
            "Added:",
            added,
        )

        print(
            "Vector count:",
            store.count(),
        )

        assert added is True
        assert store.count() == 1

        print("\n[3] Adding face:102...")

        added = store.add(
            embedding=embedding_b,
            vector_id="face:102",
            cluster_id=2,
        )

        print(
            "Added:",
            added,
        )

        print(
            "Vector count:",
            store.count(),
        )

        assert added is True
        assert store.count() == 2

        # ----------------------------------------------------------
        # IDEMPOTENCY
        # ----------------------------------------------------------

        print(
            "\n[4] Testing duplicate insertion..."
        )

        added = store.add(
            embedding=embedding_a,
            vector_id="face:101",
            cluster_id=1,
        )

        print(
            "Duplicate insertion result:",
            added,
        )

        assert added is False
        assert store.count() == 2

        # ----------------------------------------------------------
        # SEARCH
        # ----------------------------------------------------------

        print(
            "\n[5] Searching using face:101..."
        )

        results = store.search(
            embedding=embedding_a,
            k=2,
        )

        print("Search results:")

        for result in results:
            print(result)

        assert len(results) >= 1

        assert results[0]["vector_id"] == "face:101"
        assert results[0]["cluster_id"] == 1

        assert results[0]["similarity"] > 0.99

        print(
            "Nearest-neighbor check: PASS"
        )

        # ----------------------------------------------------------
        # PERSISTENCE
        # ----------------------------------------------------------

        print(
            "\n[6] Checking persistence..."
        )

        assert index_path.exists()
        assert metadata_path.exists()

        print(
            "index.bin exists:",
            index_path.exists(),
        )

        print(
            "metadata.json exists:",
            metadata_path.exists(),
        )

        # ----------------------------------------------------------
        # RELOAD
        # ----------------------------------------------------------

        print(
            "\n[7] Creating fresh HNSWVectorStore..."
        )

        restored = HNSWVectorStore(
            index_path=index_path,
            metadata_path=metadata_path,
            dimension=512,
        )

        print(
            "Restored vector count:",
            restored.count(),
        )

        assert restored.count() == 2

        print(
            "Reload count check: PASS"
        )

        # ----------------------------------------------------------
        # SEARCH AFTER RELOAD
        # ----------------------------------------------------------

        print(
            "\n[8] Searching restored index..."
        )

        restored_results = restored.search(
            embedding=embedding_a,
            k=2,
        )

        print(
            "Restored search results:"
        )

        for result in restored_results:
            print(result)

        assert len(restored_results) >= 1

        assert (
            restored_results[0]["vector_id"]
            == "face:101"
        )

        assert (
            restored_results[0]["cluster_id"]
            == 1
        )

        assert (
            restored_results[0]["similarity"]
            > 0.99
        )

        print(
            "Reload search check: PASS"
        )

        print("\n" + "=" * 70)
        print(
            "WORKFLOW 2 HNSW VECTOR STORE: PASS"
        )
        print("=" * 70)

    finally:
        shutil.rmtree(
            temp_dir,
            ignore_errors=True,
        )


if __name__ == "__main__":
    main()