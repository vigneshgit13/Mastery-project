from __future__ import annotations

import numpy as np

from app.clustering.hnsw_store import HNSWVectorStore


def main():

    print("=" * 70)
    print("WORKFLOW 2 HNSW VECTOR RETRIEVAL TEST")
    print("=" * 70)

    store = HNSWVectorStore(
        index_path="data/hnsw/retrieval_test/index.bin",
        metadata_path="data/hnsw/retrieval_test/metadata.json",
        dimension=512,
    )

    # --------------------------------------------------------------
    # Create deterministic test embedding.
    # --------------------------------------------------------------

    rng = np.random.default_rng(42)

    original = rng.random(
        512,
        dtype=np.float32,
    )

    original = original / np.linalg.norm(
        original
    )

    print()
    print("[1] Adding test embedding...")

    added = store.add(
        embedding=original,
        vector_id="face:test-retrieval",
        cluster_id=1,
    )

    print(
        f"Added: {added}"
    )

    assert added is True

    print(
        "Insertion check: PASS"
    )

    # --------------------------------------------------------------
    # Retrieve vector.
    # --------------------------------------------------------------

    print()
    print("[2] Retrieving stored embedding...")

    restored = store.get_embedding(
        "face:test-retrieval"
    )

    assert restored is not None

    print(
        f"Shape: {restored.shape}"
    )

    print(
        f"Dtype: {restored.dtype}"
    )

    print(
        f"Norm: {np.linalg.norm(restored):.8f}"
    )

    assert restored.shape == (
        512,
    )

    assert restored.dtype == np.float32

    print(
        "Shape/dtype check: PASS"
    )

    # --------------------------------------------------------------
    # Check normalization.
    # --------------------------------------------------------------

    print()
    print("[3] Checking normalization...")

    restored_norm = np.linalg.norm(
        restored
    )

    assert np.isclose(
        restored_norm,
        1.0,
        atol=1e-4,
    )

    print(
        "Normalization check: PASS"
    )

    # --------------------------------------------------------------
    # Check vector identity.
    # --------------------------------------------------------------

    print()
    print("[4] Checking vector identity...")

    similarity = float(
        np.dot(
            original,
            restored,
        )
    )

    distance = float(
        np.linalg.norm(
            original - restored
        )
    )

    print(
        f"Cosine similarity: {similarity:.8f}"
    )

    print(
        f"L2 difference: {distance:.8f}"
    )

    assert np.isclose(
        similarity,
        1.0,
        atol=1e-5,
    )

    assert np.isclose(
        distance,
        0.0,
        atol=1e-5,
    )

    print(
        "Vector identity check: PASS"
    )

    # --------------------------------------------------------------
    # Missing vector.
    # --------------------------------------------------------------

    print()
    print("[5] Checking missing vector...")

    missing = store.get_embedding(
        "face:does-not-exist"
    )

    assert missing is None

    print(
        "Missing-vector check: PASS"
    )

    print()
    print("=" * 70)
    print(
        "WORKFLOW 2 HNSW VECTOR RETRIEVAL: PASS"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()