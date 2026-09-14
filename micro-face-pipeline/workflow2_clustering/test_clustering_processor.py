
from __future__ import annotations

"""
Workflow 2 - Clustering Processor End-to-End Integration Test

Architecture under test
------------------------

Workflow 1 HNSW
      |
      | 512-D ArcFace vectors
      v
Workflow1VectorReader
      |
      v
ClusteringService
      |
      +----------------------+
      |                      |
      v                      v
Workflow 2 HNSW          PostgreSQL
active/search store      clusters + memberships
      ^
      |
ClusteringProcessor
      ^
      |
FACE_EXTRACTION_COMPLETED


Important architectural rules
-----------------------------

1. Workflow 1 HNSW is READ-ONLY.
2. PostgreSQL remains authoritative for embedding metadata/state.
3. Workflow 2 HNSW is an independent active clustering/search store.
4. ClusteringProcessor owns event/image orchestration only.
5. ClusteringService owns similarity search and cluster assignment.
6. The processor retrieves embeddings by image_id through:
       repository.get_face_embeddings_for_image(image_id)
7. PostgreSQL clustering state is reset before this test.
8. Workflow 1 faces/embeddings are NEVER deleted.
9. A unique Workflow 2 HNSW directory is created for every run.
10. The test does not require a fixed total number of Workflow 1
    embeddings. The database may legitimately contain 46, 51, etc.
11. Idempotency is verified by processing the same completion event twice.
12. Identical vectors are allowed to produce nearest-neighbor ties.
"""


import gc
import inspect
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

import faiss
from sqlalchemy import text

from app.clustering.clustering_service import ClusteringService
from app.clustering.hnsw_store import HNSWVectorStore
from app.clustering.vector_bootstrap import Workflow1VectorReader
from app.db.clustering_repository import ClusteringRepository
from app.db.postgres import SessionLocal
from app.clustering.clustering_processor import ClusteringProcessor
from app.models.face_extraction_completed import (
    FaceExtractionCompletedEvent,
)


# ======================================================================
# CONSTANTS
# ======================================================================

EXPECTED_DIMENSION = 512
SIMILARITY_THRESHOLD = 0.50
SEARCH_K = 5

PROJECT_ROOT = Path(__file__).resolve().parent

# ----------------------------------------------------------------------
# Workflow 1 HNSW
# ----------------------------------------------------------------------

WORKFLOW1_ROOT = (
    PROJECT_ROOT.parent
    / "workflow1_worker"
)

WORKFLOW1_HNSW_ROOT = (
    WORKFLOW1_ROOT
    / "data"
    / "hnsw"
)

WORKFLOW1_INDEX = (
    WORKFLOW1_HNSW_ROOT
    / "index.bin"
)

WORKFLOW1_METADATA = (
    WORKFLOW1_HNSW_ROOT
    / "metadata.json"
)

# ----------------------------------------------------------------------
# Workflow 2 test HNSW
#
# IMPORTANT:
# A unique directory is used for every run.
#
# This avoids Windows native HNSW/FAISS file-lock problems caused by
# reusing a directory from a previous failed process.
# ----------------------------------------------------------------------

WORKFLOW2_HNSW_BASE = (
    PROJECT_ROOT
    / "data"
    / "hnsw"
)

RUN_ID = (
    f"{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    f"_{__import__('os').getpid()}"
    f"_{uuid.uuid4().hex[:8]}"
)

WORKFLOW2_TEST_ROOT = (
    WORKFLOW2_HNSW_BASE
    / f"processor_test_{RUN_ID}"
)

WORKFLOW2_TEST_INDEX = (
    WORKFLOW2_TEST_ROOT
    / "index.bin"
)

WORKFLOW2_TEST_METADATA = (
    WORKFLOW2_TEST_ROOT
    / "metadata.json"
)


# ======================================================================
# OUTPUT / FAILURE HELPERS
# ======================================================================

def fail(message: str) -> None:
    """
    Fail the integration test with a clear AssertionError.
    """
    raise AssertionError(message)


def _record_value(
    record: Any,
    name: str,
) -> Any:
    """
    Read a field from either:

        - a dataclass/object
        - a dictionary-like record

    This keeps the test tolerant of the project's current
    FaceEmbeddingRecord representation without modifying production code.
    """

    if isinstance(record, dict):
        return record.get(name)

    return getattr(record, name, None)


# ======================================================================
# HNSW TEST DIRECTORY
# ======================================================================

def prepare_test_hnsw() -> None:
    """
    Create a completely isolated Workflow 2 HNSW directory.

    We intentionally do NOT delete any previous processor_test_* directory.
    This prevents Windows native HNSW/FAISS file locking from affecting
    subsequent test runs.
    """

    WORKFLOW2_TEST_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Processor test HNSW path: "
        f"{WORKFLOW2_TEST_ROOT}"
    )

    print(
        "Processor test HNSW state: PASS"
    )


def cleanup_test_hnsw() -> None:
    """
    Best-effort cleanup of the isolated Workflow 2 test directory.

    Cleanup failure must never invalidate an otherwise successful test,
    especially on Windows where native FAISS/HNSW handles can sometimes
    remain locked briefly.
    """

    if not WORKFLOW2_TEST_ROOT.exists():
        return

    try:
        shutil.rmtree(
            WORKFLOW2_TEST_ROOT
        )

        print(
            "Processor test HNSW cleanup: PASS"
        )

    except PermissionError:
        print(
            "Processor test HNSW cleanup: WARNING"
        )
        print(
            "Windows still has the temporary HNSW "
            "files locked."
        )
        print(
            f"Temporary directory retained:\n"
            f"{WORKFLOW2_TEST_ROOT}"
        )

    except Exception as exc:
        print(
            "Processor test HNSW cleanup: WARNING"
        )
        print(
            f"Cleanup error: {exc}"
        )


# ======================================================================
# WORKFLOW 2 POSTGRESQL RESET
# ======================================================================

def reset_workflow2_state(
    repository: ClusteringRepository,
) -> None:
    """
    Reset only Workflow 2 clustering state.

    This MUST NOT delete:

        face_pipeline.faces
        face_pipeline.face_embeddings

    Those are Workflow 1 authoritative data.

    It only resets:

        face_pipeline.face_cluster_members
        face_pipeline.face_clusters
    """

    print()
    print(
        "[0.1] Resetting Workflow 2 PostgreSQL clustering state..."
    )

    repository.reset_clustering_state()

    print(
        "PostgreSQL clustering state reset: PASS"
    )


# ======================================================================
# INITIAL POSTGRESQL STATE
# ======================================================================

def verify_initial_postgres_state(
    repository: ClusteringRepository,
) -> int:
    """
    Verify that Workflow 1 has persisted embeddings and Workflow 2 starts
    without cluster state.

    IMPORTANT:
    We intentionally do NOT expect a fixed embedding count such as 20.

    The current database may contain 46 legitimate embeddings.
    """

    print()
    print(
        "[0.2] Checking initial PostgreSQL state..."
    )

    embedding_count = (
        repository.count_embeddings()
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

    # --------------------------------------------------------------
    # Workflow 1 data must exist.
    # --------------------------------------------------------------

    if embedding_count <= 0:
        fail(
            "Workflow 2 requires persisted Workflow 1 embeddings. "
            f"Actual={embedding_count}"
        )

    # --------------------------------------------------------------
    # Workflow 2 must start clean.
    # --------------------------------------------------------------

    if cluster_count != 0:
        fail(
            "Workflow 2 must start with zero clusters. "
            f"Found={cluster_count}"
        )

    if membership_count != 0:
        fail(
            "Workflow 2 must start with zero memberships. "
            f"Found={membership_count}"
        )

    print(
        "Initial PostgreSQL state: PASS"
    )

    return embedding_count


# ======================================================================
# WORKFLOW 1 HNSW VALIDATION
# ======================================================================

def verify_workflow1_hnsw(
    repository: ClusteringRepository,
) -> tuple[
    Workflow1VectorReader,
    int,
]:
    """
    Validate the real Workflow 1 HNSW source.

    Workflow 1 HNSW is READ-ONLY.

    We validate the actual FAISS index directly because that is the
    authoritative physical vector store.

    Workflow1VectorReader is then used by ClusteringService to retrieve
    vectors by application-level vector_id.
    """

    print()
    print(
        "[1] Checking Workflow 1 HNSW..."
    )

    print(
        f"Workflow 1 index: "
        f"{WORKFLOW1_INDEX}"
    )

    print(
        f"Workflow 1 metadata: "
        f"{WORKFLOW1_METADATA}"
    )

    if not WORKFLOW1_INDEX.exists():
        fail(
            "Workflow 1 HNSW index does not exist:\n"
            f"{WORKFLOW1_INDEX}"
        )

    if not WORKFLOW1_METADATA.exists():
        fail(
            "Workflow 1 HNSW metadata does not exist:\n"
            f"{WORKFLOW1_METADATA}"
        )

    # --------------------------------------------------------------
    # Direct FAISS validation
    # --------------------------------------------------------------

    faiss_index = faiss.read_index(
        str(WORKFLOW1_INDEX)
    )

    faiss_count = int(
        faiss_index.ntotal
    )

    faiss_dimension = int(
        faiss_index.d
    )

    print(
        f"Workflow 1 FAISS vectors: "
        f"{faiss_count}"
    )

    print(
        f"Workflow 1 FAISS dimension: "
        f"{faiss_dimension}"
    )

    if faiss_count <= 0:
        fail(
            "Workflow 1 FAISS index contains no vectors."
        )

    if faiss_dimension != EXPECTED_DIMENSION:
        fail(
            "Workflow 1 FAISS dimension mismatch. "
            f"Expected={EXPECTED_DIMENSION}, "
            f"Actual={faiss_dimension}"
        )

    # --------------------------------------------------------------
    # Metadata validation
    # --------------------------------------------------------------

    with WORKFLOW1_METADATA.open(
        "r",
        encoding="utf-8",
    ) as handle:
        metadata = json.load(handle)

    vector_ids = (
        metadata.get("vector_ids")
    )

    if not isinstance(
        vector_ids,
        dict,
    ):
        fail(
            "Workflow 1 metadata does not contain "
            "a valid vector_ids mapping."
        )

    metadata_vector_count = len(
        vector_ids
    )

    print(
        f"Workflow 1 metadata vectors: "
        f"{metadata_vector_count}"
    )

    if metadata_vector_count != faiss_count:
        fail(
            "Workflow 1 metadata/vector count mismatch. "
            f"FAISS={faiss_count}, "
            f"Metadata={metadata_vector_count}"
        )

    print(
        "Workflow 1 HNSW source: PASS"
    )

    # --------------------------------------------------------------
    # Workflow 1 VectorReader
    # --------------------------------------------------------------

    source = Workflow1VectorReader(
        index_path=str(
            WORKFLOW1_INDEX
        ),
        metadata_path=str(
            WORKFLOW1_METADATA
        ),
        dimension=EXPECTED_DIMENSION,
    )

    reader_count = int(
        source.count()
    )

    print(
        f"Workflow 1 VectorReader vectors: "
        f"{reader_count}"
    )

    if reader_count != faiss_count:
        fail(
            "Workflow1VectorReader count does not match "
            "the actual FAISS index. "
            f"Reader={reader_count}, "
            f"FAISS={faiss_count}"
        )

    print(
        "Workflow 1 VectorReader: PASS"
    )

    return (
        source,
        faiss_count,
    )


# ======================================================================
# WORKFLOW 1 HNSW ↔ POSTGRESQL COUNT CROSS-CHECK
# ======================================================================

def verify_workflow1_postgres_consistency(
    repository: ClusteringRepository,
    source: Workflow1VectorReader,
    image_id: int,
    expected_face_count: int,
) -> None:
    """
    Verify Workflow 1 HNSW consistency for the image being processed.

    PostgreSQL is authoritative for the image -> face -> embedding
    relationship.

    Workflow 1 HNSW is authoritative for the actual 512-D vectors.

    IMPORTANT:
    We intentionally do NOT require:

        total HNSW vectors == total PostgreSQL embeddings

    because the current development HNSW may contain stale vectors from
    an earlier processing incarnation.

    Instead, every embedding belonging to the selected image must have
    a corresponding valid vector in Workflow 1 HNSW.
    """

    print()
    print(
        "[1.1] Cross-checking selected image "
        "Workflow 1 HNSW against PostgreSQL..."
    )

    records = repository.get_face_embeddings_for_image(
        image_id
    )

    if len(records) != expected_face_count:
        fail(
            "Selected image embedding count changed unexpectedly. "
            f"Expected={expected_face_count}, "
            f"Actual={len(records)}, "
            f"image_id={image_id}"
        )

    if not records:
        fail(
            "Selected image has no PostgreSQL embeddings. "
            f"image_id={image_id}"
        )

    missing_vector_ids = []

    for record in records:
        vector_id = _record_value(
            record,
            "vector_id",
        )

        if not vector_id:
            fail(
                "PostgreSQL embedding has no vector_id. "
                f"image_id={image_id}"
            )

        try:
            vector = source.get_embedding(
                str(vector_id)
            )
        except Exception as exc:
            fail(
                "Workflow 1 HNSW vector could not be read. "
                f"vector_id={vector_id}, "
                f"image_id={image_id}, "
                f"error={exc}"
            )

        if vector is None:
            missing_vector_ids.append(
                str(vector_id)
            )
            continue

        if len(vector) != EXPECTED_DIMENSION:
            fail(
                "Workflow 1 HNSW vector dimension mismatch. "
                f"vector_id={vector_id}, "
                f"Expected={EXPECTED_DIMENSION}, "
                f"Actual={len(vector)}"
            )

    if missing_vector_ids:
        fail(
            "Selected image has PostgreSQL embeddings whose "
            "vectors are missing from Workflow 1 HNSW. "
            f"image_id={image_id}, "
            f"missing={missing_vector_ids}"
        )

    print(
        f"Selected image_id: {image_id}"
    )

    print(
        f"Selected PostgreSQL embeddings: "
        f"{len(records)}"
    )

    print(
        "Selected-image PostgreSQL ↔ "
        "Workflow 1 HNSW: PASS"
    )


# ======================================================================
# IMAGE DISCOVERY
# ======================================================================

def select_test_image(
    repository: ClusteringRepository,
) -> tuple[
    int,
    int,
    int | None,
    str | None,
    str | None,
]:
    """
    Find the first real image that has persisted face embeddings.

    IMPORTANT:

    FaceEmbeddingRecord intentionally does not contain image_id.

    Therefore image_id is discovered from the authoritative PostgreSQL
    relationship:

        uploads
            |
        images
            |
          faces
            |
      face_embeddings

    SQLAlchemy is used because repository.db is a SQLAlchemy Session.
    """

    print()
    print(
        "[2] Selecting one real image..."
    )

    query = text(
        """
        SELECT
            i.id AS image_id,
            i.upload_id AS upload_id,
            i.bucket AS bucket,
            i.blob_name AS blob_name,
            COUNT(fe.face_id) AS embedding_count
        FROM face_pipeline.images i
        INNER JOIN face_pipeline.faces f
            ON f.image_id = i.id
        INNER JOIN face_pipeline.face_embeddings fe
            ON fe.face_id = f.id
        GROUP BY
            i.id,
            i.upload_id,
            i.bucket,
            i.blob_name
        HAVING COUNT(fe.face_id) > 0
        ORDER BY i.id
        LIMIT 1
        """
    )

    result = (
        repository.db.execute(
            query
        )
    )

    row = (
        result.mappings().first()
    )

    if row is None:
        fail(
            "No image with persisted face embeddings "
            "was found in PostgreSQL."
        )

    image_id = int(
        row["image_id"]
    )

    upload_id = (
        int(row["upload_id"])
        if row["upload_id"] is not None
        else None
    )

    bucket = (
        str(row["bucket"])
        if row["bucket"] is not None
        else None
    )

    blob_name = (
        str(row["blob_name"])
        if row["blob_name"] is not None
        else None
    )

    # --------------------------------------------------------------
    # IMPORTANT:
    # Use the repository's authoritative image-scoped API.
    # --------------------------------------------------------------

    records = (
        repository.get_face_embeddings_for_image(
            image_id
        )
    )

    face_count = len(
        records
    )

    if face_count <= 0:
        fail(
            "Selected image has no embedding records. "
            f"image_id={image_id}"
        )

    # --------------------------------------------------------------
    # Cross-check SQL count against repository result.
    # --------------------------------------------------------------

    sql_embedding_count = int(
        row["embedding_count"]
    )

    if sql_embedding_count != face_count:
        fail(
            "Image embedding count mismatch. "
            f"SQL={sql_embedding_count}, "
            f"Repository={face_count}, "
            f"image_id={image_id}"
        )

    print(
        f"Selected image_id: {image_id}"
    )

    print(
        f"Expected faces: {face_count}"
    )

    if upload_id is not None:
        print(
            f"Upload id: {upload_id}"
        )

    if bucket is not None:
        print(
            f"Bucket: {bucket}"
        )

    if blob_name is not None:
        print(
            f"Blob: {blob_name}"
        )

    print(
        "Image-scoped PostgreSQL lookup: PASS"
    )

    return (
        image_id,
        face_count,
        upload_id,
        bucket,
        blob_name,
    )


# ======================================================================
# EVENT CREATION
# ======================================================================

def _value_for_event_parameter(
    name: str,
    *,
    event_id: uuid.UUID,
    upload_id: int,
    image_id: int,
    bucket: str | None,
    blob_name: str | None,
    face_count: int,
) -> Any:
    """
    Resolve constructor arguments for the currently installed
    FaceExtractionCompletedEvent model.

    The test introspects the actual constructor so it does not assume
    an obsolete signature.
    """

    normalized = (
        name.lower()
    )

    if normalized in {
        "event_id",
        "id",
    }:
        return event_id

    if normalized in {
        "upload_id",
        "upload",
    }:
        return upload_id

    if normalized in {
        "image_id",
        "image",
    }:
        return image_id

    if normalized in {
        "bucket",
        "bucket_name",
    }:
        return (
            bucket
            if bucket is not None
            else "test-face-clustering"
        )

    if normalized in {
        "blob_name",
        "object_name",
        "gcs_blob_name",
    }:
        return (
            blob_name
            if blob_name is not None
            else f"test/image-{image_id}.jpg"
        )

    if normalized in {
        "face_count",
        "faces",
        "number_of_faces",
    }:
        return face_count

    if normalized in {
        "event_type",
        "type",
    }:
        return "FACE_EXTRACTION_COMPLETED"

    if normalized in {
        "event_version",
        "version",
    }:
        return "1.0"

    return None


def create_completion_event(
    *,
    upload_id: int,
    image_id: int,
    face_count: int,
    bucket: str | None,
    blob_name: str | None,
):
    """
    Construct the installed FaceExtractionCompletedEvent model.

    Only constructor parameters that are actually present are supplied.
    """

    print()
    print(
        "[3] Creating FACE_EXTRACTION_COMPLETED event..."
    )

    event_id = uuid.uuid4()

    signature = inspect.signature(
        FaceExtractionCompletedEvent
    )

    kwargs: dict[str, Any] = {}

    for name, parameter in (
        signature.parameters.items()
    ):
        if name == "self":
            continue

        value = _value_for_event_parameter(
            name,
            event_id=event_id,
            upload_id=upload_id,
            image_id=image_id,
            bucket=bucket,
            blob_name=blob_name,
            face_count=face_count,
        )

        if value is not None:
            kwargs[name] = value

    event = FaceExtractionCompletedEvent(
        **kwargs
    )

    print(
        f"event_id: {event.event_id}"
    )

    print(
        f"image_id: {event.image_id}"
    )

    print(
        f"face_count: {event.face_count}"
    )

    print(
        "Completion event creation: PASS"
    )

    return event


# ======================================================================
# EVENT VALIDATION
# ======================================================================

def verify_event(
    event,
    image_id: int,
    face_count: int,
) -> None:
    """
    Verify the event matches the selected image.
    """

    print()
    print(
        "[3.1] Validating completion event..."
    )

    if int(event.image_id) != int(
        image_id
    ):
        fail(
            "Event image_id mismatch. "
            f"Expected={image_id}, "
            f"Actual={event.image_id}"
        )

    if int(event.face_count) != int(
        face_count
    ):
        fail(
            "Event face_count mismatch. "
            f"Expected={face_count}, "
            f"Actual={event.face_count}"
        )

    event.validate_event_type()

    print(
        f"Event image_id: {event.image_id}"
    )

    print(
        f"Event face_count: {event.face_count}"
    )

    print(
        "Completion event validation: PASS"
    )


# ======================================================================
# WORKFLOW 2 HNSW CREATION
# ======================================================================

def create_empty_workflow2_hnsw() -> HNSWVectorStore:
    """
    Create a fresh empty Workflow 2 HNSW store.
    """

    print()
    print(
        "[4] Creating clean Workflow 2 HNSW..."
    )

    target = HNSWVectorStore(
        index_path=str(
            WORKFLOW2_TEST_INDEX
        ),
        metadata_path=str(
            WORKFLOW2_TEST_METADATA
        ),
        dimension=EXPECTED_DIMENSION,
    )

    target_count = int(
        target.count()
    )

    print(
        f"Workflow 2 HNSW vectors: "
        f"{target_count}"
    )

    if target_count != 0:
        fail(
            "Workflow 2 HNSW must start empty. "
            f"Actual={target_count}"
        )

    print(
        "Workflow 2 HNSW empty: PASS"
    )

    return target


# ======================================================================
# FIRST PROCESSING RESULT
# ======================================================================

def verify_first_processing_result(
    result,
    face_count: int,
) -> None:
    """
    Validate the ClusteringProcessor result for the first event.
    """

    print()
    print(
        "[7] Verifying processor result..."
    )

    print(
        f"Status: {result.status}"
    )

    print(
        f"Processed faces: "
        f"{result.processed_face_count}"
    )

    print(
        f"Created clusters: "
        f"{result.created_cluster_count}"
    )

    print(
        f"Existing clusters: "
        f"{result.existing_cluster_count}"
    )

    print(
        f"Assignments: "
        f"{len(result.assignments)}"
    )

    if result.status != "COMPLETED":
        fail(
            "Processor did not complete successfully. "
            f"Status={result.status}"
        )

    if (
        result.processed_face_count
        != face_count
    ):
        fail(
            "Processor face count mismatch. "
            f"Expected={face_count}, "
            f"Actual={result.processed_face_count}"
        )

    if (
        result.created_cluster_count
        + result.existing_cluster_count
        != face_count
    ):
        fail(
            "Cluster assignment count mismatch. "
            f"Created={result.created_cluster_count}, "
            f"Existing={result.existing_cluster_count}, "
            f"Expected={face_count}"
        )

    if len(
        result.assignments
    ) != face_count:
        fail(
            "Assignment result length mismatch. "
            f"Expected={face_count}, "
            f"Actual={len(result.assignments)}"
        )

    print(
        "Processor execution: PASS"
    )


# ======================================================================
# POSTGRESQL VERIFICATION
# ======================================================================

def verify_postgres_after_processing(
    repository: ClusteringRepository,
    face_count: int,
) -> None:
    """
    Verify Workflow 2 PostgreSQL cluster/member state.
    """

    print()
    print(
        "[8] Verifying PostgreSQL clustering state..."
    )

    cluster_count = (
        repository.count_clusters()
    )

    membership_count = (
        repository.count_memberships()
    )

    print(
        f"Clusters: {cluster_count}"
    )

    print(
        f"Memberships: {membership_count}"
    )

    if cluster_count <= 0:
        fail(
            "No Workflow 2 clusters were created."
        )

    if membership_count != face_count:
        fail(
            "Unexpected membership count. "
            f"Expected={face_count}, "
            f"Actual={membership_count}"
        )

    print(
        "PostgreSQL clustering state: PASS"
    )


# ======================================================================
# IMAGE-SCOPED ASSIGNMENT VERIFICATION
# ======================================================================

def verify_image_assignments(
    repository: ClusteringRepository,
    image_id: int,
    face_count: int,
) -> list[Any]:
    """
    Verify that every face belonging to the selected image has a cluster.

    We intentionally use:

        get_face_embeddings_for_image(image_id)

    rather than assuming FaceEmbeddingRecord contains image_id.
    """

    print()
    print(
        "[8.1] Verifying image-scoped assignments..."
    )

    records = (
        repository.get_face_embeddings_for_image(
            image_id
        )
    )

    if len(records) != face_count:
        fail(
            "Image-scoped embedding count changed unexpectedly. "
            f"Expected={face_count}, "
            f"Actual={len(records)}"
        )

    face_ids: list[int] = []

    for record in records:
        face_id_value = _record_value(
            record,
            "face_id",
        )

        if face_id_value is None:
            fail(
                "Embedding record does not contain face_id."
            )

        face_id = int(
            face_id_value
        )

        face_ids.append(
            face_id
        )

        cluster_id = (
            repository.get_cluster_for_face(
                face_id
            )
        )

        if cluster_id is None:
            fail(
                "Selected image face has no cluster membership. "
                f"image_id={image_id}, "
                f"face_id={face_id}"
            )

    if len(
        set(face_ids)
    ) != face_count:
        fail(
            "Duplicate face IDs found in image-scoped records. "
            f"Expected unique={face_count}, "
            f"Actual unique={len(set(face_ids))}"
        )

    print(
        f"All {face_count} faces for image "
        f"{image_id} have cluster memberships: PASS"
    )

    return records


# ======================================================================
# WORKFLOW 2 HNSW VERIFICATION
# ======================================================================

def verify_workflow2_hnsw(
    target: HNSWVectorStore,
    face_count: int,
) -> None:
    """
    Verify that exactly the selected image's faces were inserted into
    Workflow 2 HNSW.
    """

    print()
    print(
        "[9] Verifying Workflow 2 HNSW..."
    )

    target_count = int(
        target.count()
    )

    print(
        f"Workflow 2 HNSW vectors: "
        f"{target_count}"
    )

    if target_count != face_count:
        fail(
            "Workflow 2 HNSW vector count mismatch. "
            f"Expected={face_count}, "
            f"Actual={target_count}"
        )

    print(
        "Workflow 2 HNSW state: PASS"
    )


# ======================================================================
# POSTGRESQL ↔ WORKFLOW 2 HNSW MAPPING
# ======================================================================

def verify_postgres_hnsw_mappings(
    repository: ClusteringRepository,
    target: HNSWVectorStore,
    records: list[Any],
) -> None:
    """
    Verify every processed face has:

        PostgreSQL cluster membership
        +
        Workflow 2 HNSW vector
        +
        matching cluster_id

    The exact nearest vector ID is NOT required to match because
    identical vectors can legitimately produce nearest-neighbor ties.
    """

    print()
    print(
        "[10] Verifying PostgreSQL/HNSW mappings..."
    )

    checked = 0

    for record in records:
        face_id_value = _record_value(
            record,
            "face_id",
        )

        vector_id_value = _record_value(
            record,
            "vector_id",
        )

        if face_id_value is None:
            fail(
                "Embedding record is missing face_id."
            )

        if vector_id_value is None:
            fail(
                "Embedding record is missing vector_id."
            )

        face_id = int(
            face_id_value
        )

        vector_id = str(
            vector_id_value
        )

        expected_cluster = (
            repository.get_cluster_for_face(
                face_id
            )
        )

        if expected_cluster is None:
            fail(
                "No PostgreSQL cluster found for face. "
                f"face_id={face_id}"
            )

        # ----------------------------------------------------------
        # Workflow 2 HNSW must know this vector.
        # ----------------------------------------------------------

        actual_cluster = (
            target.get_cluster_id(
                vector_id
            )
        )

        if actual_cluster is None:
            fail(
                "Workflow 2 HNSW does not contain expected vector. "
                f"vector_id={vector_id}"
            )

        if int(actual_cluster) != int(
            expected_cluster
        ):
            fail(
                "PostgreSQL/HNSW cluster mismatch. "
                f"face_id={face_id}, "
                f"vector_id={vector_id}, "
                f"PostgreSQL_cluster={expected_cluster}, "
                f"HNSW_cluster={actual_cluster}"
            )

        checked += 1

    print(
        f"Verified {checked} PostgreSQL/HNSW mappings."
    )

    print(
        "PostgreSQL/HNSW cluster mappings: PASS"
    )


# ======================================================================
# PROCESSOR IDEMPOTENCY
# ======================================================================

def verify_processor_idempotency(
    processor: ClusteringProcessor,
    event,
    repository: ClusteringRepository,
    target: HNSWVectorStore,
    face_count: int,
) -> None:
    """
    Process the exact same completion event a second time.

    Expected:

        processed_face_count = face_count
        created_cluster_count = 0
        existing_cluster_count = face_count

    and:

        cluster count unchanged
        membership count unchanged
        HNSW vector count unchanged
    """

    print()
    print(
        "[11] Testing processor idempotency..."
    )

    clusters_before = (
        repository.count_clusters()
    )

    memberships_before = (
        repository.count_memberships()
    )

    hnsw_before = int(
        target.count()
    )

    print(
        f"Before rerun - clusters: "
        f"{clusters_before}"
    )

    print(
        f"Before rerun - memberships: "
        f"{memberships_before}"
    )

    print(
        f"Before rerun - HNSW vectors: "
        f"{hnsw_before}"
    )

    # --------------------------------------------------------------
    # Replay the exact same event.
    # --------------------------------------------------------------

    second_result = (
        processor.process(
            event
        )
    )

    print(
        f"Second status: "
        f"{second_result.status}"
    )

    print(
        f"Second processed faces: "
        f"{second_result.processed_face_count}"
    )

    print(
        f"Second created clusters: "
        f"{second_result.created_cluster_count}"
    )

    print(
        f"Second existing clusters: "
        f"{second_result.existing_cluster_count}"
    )

    if second_result.status != "COMPLETED":
        fail(
            "Second processor execution failed. "
            f"Status={second_result.status}"
        )

    if (
        second_result.processed_face_count
        != face_count
    ):
        fail(
            "Second processing face count mismatch. "
            f"Expected={face_count}, "
            f"Actual={second_result.processed_face_count}"
        )

    if second_result.created_cluster_count != 0:
        fail(
            "Idempotent rerun created new clusters. "
            f"Created={second_result.created_cluster_count}"
        )

    if (
        second_result.existing_cluster_count
        != face_count
    ):
        fail(
            "Idempotent rerun did not recognize all "
            "faces as existing memberships. "
            f"Expected={face_count}, "
            f"Actual={second_result.existing_cluster_count}"
        )

    # --------------------------------------------------------------
    # Verify database state did not change.
    # --------------------------------------------------------------

    clusters_after = (
        repository.count_clusters()
    )

    memberships_after = (
        repository.count_memberships()
    )

    hnsw_after = int(
        target.count()
    )

    print(
        f"After rerun - clusters: "
        f"{clusters_after}"
    )

    print(
        f"After rerun - memberships: "
        f"{memberships_after}"
    )

    print(
        f"After rerun - HNSW vectors: "
        f"{hnsw_after}"
    )

    if clusters_after != clusters_before:
        fail(
            "Idempotent rerun changed cluster count. "
            f"Before={clusters_before}, "
            f"After={clusters_after}"
        )

    if memberships_after != memberships_before:
        fail(
            "Idempotent rerun changed membership count. "
            f"Before={memberships_before}, "
            f"After={memberships_after}"
        )

    if hnsw_after != hnsw_before:
        fail(
            "Idempotent rerun changed Workflow 2 HNSW vector count. "
            f"Before={hnsw_before}, "
            f"After={hnsw_after}"
        )

    print(
        "Processor idempotency: PASS"
    )


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    """
    Execute the complete Workflow 2 ClusteringProcessor integration test.
    """

    print("=" * 70)
    print(
        "WORKFLOW 2 CLUSTERING PROCESSOR TEST"
    )
    print("=" * 70)

    db = None
    repository = None
    source = None
    target = None
    clustering_service = None
    processor = None

    try:
        # ==============================================================
        # [0] TEST HNSW ISOLATION
        # ==============================================================

        print()
        print(
            "[0] Creating isolated Workflow 2 test HNSW..."
        )

        prepare_test_hnsw()

        # ==============================================================
        # [0.1] DATABASE / REPOSITORY
        # ==============================================================

        db = SessionLocal()

        repository = ClusteringRepository(
            db
        )

        # ==============================================================
        # [0.2] RESET WORKFLOW 2 STATE
        # ==============================================================

        reset_workflow2_state(
            repository
        )

        # ==============================================================
        # [0.3] INITIAL DATABASE STATE
        # ==============================================================

        embedding_count = (
            verify_initial_postgres_state(
                repository
            )
        )

        # ==============================================================
        # [1] WORKFLOW 1 HNSW
        # ==============================================================

        (
            source,
            source_count,
        ) = verify_workflow1_hnsw(
            repository
        )

        # ==============================================================
        # [2] POSTGRESQL EMBEDDINGS
        # ==============================================================

        print()
        print(
            "[2] Checking PostgreSQL embeddings..."
        )

        current_embedding_count = (
            repository.count_embeddings()
        )

        print(
            f"PostgreSQL embeddings: "
            f"{current_embedding_count}"
        )

        if current_embedding_count <= 0:
            fail(
                "PostgreSQL contains no Workflow 1 embeddings."
            )

        if current_embedding_count != embedding_count:
            fail(
                "PostgreSQL embedding count changed unexpectedly "
                "before processing. "
                f"Initial={embedding_count}, "
                f"Current={current_embedding_count}"
            )

        print(
            "PostgreSQL embedding state: PASS"
        )

        # ==============================================================
        # [2.1] SELECT ONE REAL IMAGE
        # ==============================================================

        (
            image_id,
            face_count,
            upload_id,
            bucket,
            blob_name,
        ) = select_test_image(
            repository
        )


        verify_workflow1_postgres_consistency(
            repository=repository,
            source=source,
            image_id=image_id,
            expected_face_count=face_count,
)

        # ==============================================================
        # [3] CREATE COMPLETION EVENT
        # ==============================================================

        if upload_id is None:
            fail(
                "Selected image does not have an upload_id."
            )

        event = create_completion_event(
            upload_id=upload_id,
            image_id=image_id,
            face_count=face_count,
            bucket=bucket,
            blob_name=blob_name,
        )

        verify_event(
            event,
            image_id,
            face_count,
        )

        # ==============================================================
        # [4] CREATE EMPTY WORKFLOW 2 HNSW
        # ==============================================================

        target = create_empty_workflow2_hnsw()

        # ==============================================================
        # [5] CREATE CLUSTERING SERVICE
        # ==============================================================

        print()
        print(
            "[5] Creating ClusteringService..."
        )

        clustering_service = ClusteringService(
            repository=repository,
            source_store=source,
            active_store=target,
            similarity_threshold=SIMILARITY_THRESHOLD,
            search_k=SEARCH_K,
        )

        print(
            "ClusteringService: PASS"
        )

        # ==============================================================
        # [6] CREATE CLUSTERING PROCESSOR
        # ==============================================================

        print()
        print(
            "[6] Creating ClusteringProcessor..."
        )

        processor = ClusteringProcessor(
            repository=repository,
            clustering_service=clustering_service,
        )

        print(
            "ClusteringProcessor: PASS"
        )

        # ==============================================================
        # [7] PROCESS REAL COMPLETION EVENT
        # ==============================================================

        print()
        print(
            "[7] Processing real completion event..."
        )

        result = processor.process(
            event
        )

        verify_first_processing_result(
            result,
            face_count,
        )

        # ==============================================================
        # [8] POSTGRESQL CLUSTERING STATE
        # ==============================================================

        verify_postgres_after_processing(
            repository,
            face_count,
        )

        # ==============================================================
        # [8.1] IMAGE-SCOPED ASSIGNMENTS
        # ==============================================================

        image_records = (
            verify_image_assignments(
                repository,
                image_id,
                face_count,
            )
        )

        # ==============================================================
        # [9] WORKFLOW 2 HNSW
        # ==============================================================

        verify_workflow2_hnsw(
            target,
            face_count,
        )

        # ==============================================================
        # [10] POSTGRESQL ↔ HNSW MAPPINGS
        # ==============================================================

        verify_postgres_hnsw_mappings(
            repository,
            target,
            image_records,
        )

        # ==============================================================
        # [11] IDEMPOTENCY
        # ==============================================================

        verify_processor_idempotency(
            processor,
            event,
            repository,
            target,
            face_count,
        )

        # ==============================================================
        # FINAL
        # ==============================================================

        print()
        print("=" * 70)
        print(
            "WORKFLOW 2 CLUSTERING PROCESSOR: PASS"
        )
        print("=" * 70)

        print()
        print(
            "Verified:"
        )

        print(
            "  PostgreSQL embeddings          : PASS"
        )

        print(
            "  Workflow 1 HNSW source        : PASS"
        )

        print(
            "  Selected-image HNSW/PG mapping: PASS"
        )

        print(
            "  Image-scoped lookup            : PASS"
        )

        print(
            "  Completion event validation    : PASS"
        )

        print(
            "  ClusteringService              : PASS"
        )

        print(
            "  ClusteringProcessor            : PASS"
        )

        print(
            "  PostgreSQL clusters            : PASS"
        )

        print(
            "  PostgreSQL memberships         : PASS"
        )

        print(
            "  Workflow 2 HNSW                : PASS"
        )

        print(
            "  PostgreSQL/HNSW mappings       : PASS"
        )

        print(
            "  Processor idempotency          : PASS"
        )

    except Exception:
        # --------------------------------------------------------------
        # Roll back only the current SQLAlchemy transaction.
        # --------------------------------------------------------------

        if repository is not None:
            try:
                repository.rollback()
            except Exception:
                pass

        raise

    finally:
        # --------------------------------------------------------------
        # IMPORTANT:
        #
        # Release references to the HNSW/native objects BEFORE attempting
        # to remove the temporary directory.
        # --------------------------------------------------------------

        processor = None
        clustering_service = None
        target = None
        source = None

        gc.collect()

        # --------------------------------------------------------------
        # Close SQLAlchemy session.
        # --------------------------------------------------------------

        if db is not None:
            try:
                db.rollback()
            except Exception:
                pass

            try:
                db.close()
            except Exception:
                pass

        # --------------------------------------------------------------
        # Give Python/FAISS a chance to release native resources.
        # --------------------------------------------------------------

        gc.collect()

        # --------------------------------------------------------------
        # Best-effort temporary HNSW cleanup.
        # --------------------------------------------------------------

        cleanup_test_hnsw()


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()