from pathlib import Path
import shutil
import tempfile

import numpy as np

from app.services.faiss_service import FaissService


def main() -> None:
    temp_dir = Path(
        tempfile.mkdtemp(prefix="workflow1_faiss_test_")
    )

    print("Temporary test directory:", temp_dir)

    try:
        print("\n[1] Creating FaissService...")

        service = FaissService(
            index_path=temp_dir / "index.bin",
            metadata_path=temp_dir / "metadata.json",
            dimension=512,
        )

        print("Service created.")
        print("Initial vector count:", service.count)

        rng = np.random.default_rng(42)

        embedding_1 = rng.normal(
            size=512
        ).astype(np.float32)

        embedding_1 /= np.linalg.norm(embedding_1)

        embedding_2 = rng.normal(
            size=512
        ).astype(np.float32)

        embedding_2 /= np.linalg.norm(embedding_2)

        print("\n[2] Adding first embedding...")

        result_1 = service.add(
            embedding=embedding_1,
            vector_id="face:101",
            metadata={
                "face_id": 101,
                "image_id": 1,
            },
        )

        print("Result:", result_1)

        print("\n[3] Adding second embedding...")

        result_2 = service.add(
            embedding=embedding_2,
            vector_id="face:102",
            metadata={
                "face_id": 102,
                "image_id": 1,
            },
        )

        print("Result:", result_2)

        print("\n[4] Checking vector count...")

        print("Vector count:", service.count)

        assert service.count == 2

        print("Vector count check: PASS")

        print("\n[5] Searching using face:101 embedding...")

        results = service.search(
            embedding=embedding_1,
            k=2,
        )

        print("Search results:")

        for result in results:
            print(result)

        assert len(results) == 2
        assert results[0]["vector_id"] == "face:101"

        print("Nearest-neighbor check: PASS")

        print("\n[6] Testing idempotent insertion...")

        duplicate = service.add(
            embedding=embedding_1,
            vector_id="face:101",
            metadata={
                "face_id": 101,
                "image_id": 1,
            },
        )

        print("Duplicate result:", duplicate)

        assert duplicate["added"] is False
        assert service.count == 2

        print("Idempotency check: PASS")

        print("\n[7] Testing persistence...")

        service.close()

        index_file = temp_dir / "index.bin"
        metadata_file = temp_dir / "metadata.json"

        assert index_file.exists()
        assert metadata_file.exists()

        print("index.bin exists:", index_file.exists())
        print("metadata.json exists:", metadata_file.exists())

        print("\n[8] Reloading FAISS service...")

        restored = FaissService(
            index_path=index_file,
            metadata_path=metadata_file,
            dimension=512,
        )

        print(
            "Restored vector count:",
            restored.count,
        )

        assert restored.count == 2

        print("Reload count check: PASS")

        print("\n[9] Searching restored index...")

        restored_results = restored.search(
            embedding=embedding_1,
            k=2,
        )

        for result in restored_results:
            print(result)

        assert restored_results
        assert restored_results[0]["vector_id"] == "face:101"

        print("Reload search check: PASS")

        print("\n" + "=" * 60)
        print("FAISS/HNSW SERVICE VERIFICATION: PASS")
        print("=" * 60)

    finally:
        shutil.rmtree(
            temp_dir,
            ignore_errors=True,
        )

        print("\nTemporary test directory removed.")


if __name__ == "__main__":
    main()