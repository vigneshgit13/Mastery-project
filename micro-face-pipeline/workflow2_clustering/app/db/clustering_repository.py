from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session


SCHEMA = "face_pipeline"


# ======================================================================
# DATA OBJECTS
# ======================================================================


@dataclass(frozen=True)
class FaceEmbeddingRecord:
    """
    PostgreSQL metadata describing one face embedding.

    IMPORTANT:
    This does NOT contain the actual 512-D vector.

    The actual vector lives in HNSW/FAISS.
    """

    face_id: int
    vector_id: str
    model_name: str
    model_version: str | None
    embedding_dimension: int
    vector_store: str
    normalized: bool
    embedding_norm: float | None


@dataclass(frozen=True)
class ClusterMemberRecord:
    """
    PostgreSQL metadata describing one face-cluster membership.
    """

    id: int
    cluster_id: int
    face_id: int
    similarity: float


# ======================================================================
# CLUSTERING REPOSITORY
# ======================================================================


class ClusteringRepository:
    """
    PostgreSQL repository for Workflow 2 clustering state.

    PostgreSQL is the authoritative source for:

        face_embeddings
        face_clusters
        face_cluster_members

    HNSW/FAISS remains responsible for:

        actual 512-D vectors
        nearest-neighbor search
    """

    def __init__(
        self,
        db: Session,
    ) -> None:
        self.db = db

    # ==================================================================
    # FACE EMBEDDING METADATA
    # ==================================================================

    def get_face_embedding(
        self,
        face_id: int,
    ) -> FaceEmbeddingRecord | None:
        """
        Retrieve embedding metadata for one face.

        The actual 512-D vector is NOT stored in PostgreSQL.
        """

        query = text(
            f"""
            SELECT
                face_id,
                vector_id,
                model_name,
                model_version,
                embedding_dimension,
                vector_store,
                normalized,
                embedding_norm
            FROM {SCHEMA}.face_embeddings
            WHERE face_id = :face_id
            """
        )

        row = (
            self.db.execute(
                query,
                {
                    "face_id": int(face_id),
                },
            )
            .mappings()
            .first()
        )

        if row is None:
            return None

        return FaceEmbeddingRecord(
            face_id=int(row["face_id"]),
            vector_id=str(row["vector_id"]),
            model_name=str(row["model_name"]),
            model_version=(
                str(row["model_version"])
                if row["model_version"] is not None
                else None
            ),
            embedding_dimension=int(
                row["embedding_dimension"]
            ),
            vector_store=str(
                row["vector_store"]
            ),
            normalized=bool(
                row["normalized"]
            ),
            embedding_norm=(
                float(row["embedding_norm"])
                if row["embedding_norm"] is not None
                else None
            ),
        )

    # ==================================================================
    # ALL FACE EMBEDDINGS
    # ==================================================================

    def get_all_face_embeddings(
        self,
    ) -> list[FaceEmbeddingRecord]:
        """
        Retrieve all face embedding metadata.

        Used during Workflow 2 bootstrap/reconstruction.
        """

        query = text(
            f"""
            SELECT
                face_id,
                vector_id,
                model_name,
                model_version,
                embedding_dimension,
                vector_store,
                normalized,
                embedding_norm
            FROM {SCHEMA}.face_embeddings
            ORDER BY face_id
            """
        )

        rows = (
            self.db.execute(query)
            .mappings()
            .all()
        )

        return [
            FaceEmbeddingRecord(
                face_id=int(row["face_id"]),
                vector_id=str(row["vector_id"]),
                model_name=str(row["model_name"]),
                model_version=(
                    str(row["model_version"])
                    if row["model_version"] is not None
                    else None
                ),
                embedding_dimension=int(
                    row["embedding_dimension"]
                ),
                vector_store=str(
                    row["vector_store"]
                ),
                normalized=bool(
                    row["normalized"]
                ),
                embedding_norm=(
                    float(row["embedding_norm"])
                    if row["embedding_norm"] is not None
                    else None
                ),
            )
            for row in rows
        ]

    # ==================================================================
    # FACE EMBEDDINGS FOR IMAGE
    # ==================================================================

    def get_face_embeddings_for_image(
        self,
        image_id: int,
    ) -> list[FaceEmbeddingRecord]:
        """
        Return embedding metadata for all faces belonging to
        one image.

        PostgreSQL remains authoritative for the relationship:

            image
              ↓
            faces
              ↓
            face_embeddings

        The actual 512-D vectors are retrieved separately from
        Workflow 1 HNSW.
        """

        query = text(
            f"""
            SELECT
                fe.face_id,
                fe.vector_id,
                fe.model_name,
                fe.model_version,
                fe.embedding_dimension,
                fe.vector_store,
                fe.normalized,
                fe.embedding_norm
            FROM {SCHEMA}.face_embeddings fe
            INNER JOIN {SCHEMA}.faces f
                ON f.id = fe.face_id
            WHERE f.image_id = :image_id
            ORDER BY f.face_index, f.id
            """
        )

        rows = (
            self.db.execute(
                query,
                {
                    "image_id": int(image_id),
                },
            )
            .mappings()
            .all()
        )

        return [
            FaceEmbeddingRecord(
                face_id=int(row["face_id"]),
                vector_id=str(row["vector_id"]),
                model_name=str(row["model_name"]),
                model_version=(
                    str(row["model_version"])
                    if row["model_version"] is not None
                    else None
                ),
                embedding_dimension=int(
                    row["embedding_dimension"]
                ),
                vector_store=str(
                    row["vector_store"]
                ),
                normalized=bool(
                    row["normalized"]
                ),
                embedding_norm=(
                    float(row["embedding_norm"])
                    if row["embedding_norm"] is not None
                    else None
                ),
            )
            for row in rows
        ]

    # ==================================================================
    # CLUSTER
    # ==================================================================

    def create_cluster(
        self,
    ) -> int:
        """
        Create a new face cluster.

        Returns:
            Newly generated cluster ID.
        """

        query = text(
            f"""
            INSERT INTO {SCHEMA}.face_clusters
                DEFAULT VALUES
            RETURNING id
            """
        )

        row = (
            self.db.execute(query)
            .mappings()
            .one()
        )

        return int(
            row["id"]
        )

    # ==================================================================
    # CLUSTER MEMBER
    # ==================================================================

    def add_cluster_member(
        self,
        *,
        cluster_id: int,
        face_id: int,
        similarity: float,
    ) -> int:
        """
        Add one face to a cluster.

        The database schema guarantees that one face can belong
        to only one cluster through:

            UNIQUE(face_id)
        """

        query = text(
            f"""
            INSERT INTO {SCHEMA}.face_cluster_members
                (
                    cluster_id,
                    face_id,
                    similarity
                )
            VALUES
                (
                    :cluster_id,
                    :face_id,
                    :similarity
                )
            RETURNING id
            """
        )

        row = (
            self.db.execute(
                query,
                {
                    "cluster_id": int(
                        cluster_id
                    ),
                    "face_id": int(
                        face_id
                    ),
                    "similarity": float(
                        similarity
                    ),
                },
            )
            .mappings()
            .one()
        )

        return int(
            row["id"]
        )

    # ==================================================================
    # EXISTING MEMBERSHIP
    # ==================================================================

    def get_cluster_for_face(
        self,
        face_id: int,
    ) -> int | None:
        """
        Return the cluster ID already assigned to a face.

        Returns None if the face has not yet been clustered.
        """

        query = text(
            f"""
            SELECT cluster_id
            FROM {SCHEMA}.face_cluster_members
            WHERE face_id = :face_id
            """
        )

        row = (
            self.db.execute(
                query,
                {
                    "face_id": int(
                        face_id
                    ),
                },
            )
            .mappings()
            .first()
        )

        if row is None:
            return None

        return int(
            row["cluster_id"]
        )

    # ==================================================================
    # EVENT IDEMPOTENCY
    # ==================================================================
         # ==================================================================
    # EVENT IDEMPOTENCY
    # ==================================================================

    def is_event_completed(
        self,
        event_id: str,
    ) -> bool:
        """
        Return True when Workflow 2 has already completed this event.

        PostgreSQL is the authoritative source for event-level
        idempotency.
        """

        query = text(
            f"""
            SELECT 1
            FROM {SCHEMA}.workflow2_processed_events
            WHERE event_id = :event_id
              AND status = 'COMPLETED'
            LIMIT 1
            """
        )

        row = (
            self.db.execute(
                query,
                {
                    "event_id": str(event_id),
                },
            )
            .first()
        )

        return row is not None

    def mark_event_completed(
        self,
        *,
        event_id: str,
        event_type: str,
        workflow: str,
        upload_id: int,
        image_id: int,
    ) -> None:
        """
        Mark a Workflow 2 event as successfully completed.

        The event_id primary key prevents duplicate event records.
        Repeated completion attempts are harmless.
        """

        query = text(
            f"""
            INSERT INTO {SCHEMA}.workflow2_processed_events
                (
                    event_id,
                    event_type,
                    workflow,
                    upload_id,
                    image_id,
                    status
                )
            VALUES
                (
                    :event_id,
                    :event_type,
                    :workflow,
                    :upload_id,
                    :image_id,
                    'COMPLETED'
                )
            ON CONFLICT (event_id) DO NOTHING
            """
        )

        self.db.execute(
            query,
            {
                "event_id": str(event_id),
                "event_type": str(event_type),
                "workflow": str(workflow),
                "upload_id": int(upload_id),
                "image_id": int(image_id),
            },
        )

    # ==================================================================
    # CLUSTER MEMBERS
    # ==================================================================

    def get_cluster_members(
        self,
        cluster_id: int,
    ) -> list[ClusterMemberRecord]:
        """
        Return all members belonging to a cluster.
        """

        query = text(
            f"""
            SELECT
                id,
                cluster_id,
                face_id,
                similarity
            FROM {SCHEMA}.face_cluster_members
            WHERE cluster_id = :cluster_id
            ORDER BY face_id
            """
        )

        rows = (
            self.db.execute(
                query,
                {
                    "cluster_id": int(
                        cluster_id
                    ),
                },
            )
            .mappings()
            .all()
        )

        return [
            ClusterMemberRecord(
                id=int(row["id"]),
                cluster_id=int(
                    row["cluster_id"]
                ),
                face_id=int(
                    row["face_id"]
                ),
                similarity=float(
                    row["similarity"]
                ),
            )
            for row in rows
        ]

    # ==================================================================
    # COUNTS
    # ==================================================================

    def count_embeddings(self) -> int:
        """
        Return the total number of face embedding metadata records.

        PostgreSQL stores embedding metadata only.
        The actual 512-D vectors remain in HNSW.
        """

        query = text(
            f"""
            SELECT COUNT(*)
            FROM {SCHEMA}.face_embeddings
            """
        )

        return int(
            self.db.execute(query).scalar_one()
        )

    def count_clusters(self) -> int:
        """
        Return the total number of clusters.
        """

        query = text(
            f"""
            SELECT COUNT(*)
            FROM {SCHEMA}.face_clusters
            """
        )

        return int(
            self.db.execute(query).scalar_one()
        )

    def count_memberships(self) -> int:
        """
        Return the total number of face-cluster memberships.
        """

        query = text(
            f"""
            SELECT COUNT(*)
            FROM {SCHEMA}.face_cluster_members
            """
        )

        return int(
            self.db.execute(query).scalar_one()
        )

    # ==================================================================
    # TEST / RESET CLUSTERING STATE
    # ==================================================================

    def reset_clustering_state(
        self,
    ) -> None:
        """
        Reset Workflow 2 clustering state.

        IMPORTANT:
        This method ONLY removes Workflow 2 clustering state.

        It does NOT delete:

            face_pipeline.faces
            face_pipeline.face_embeddings

        It deletes:

            face_pipeline.face_cluster_members
            face_pipeline.face_clusters

        This is intended for deterministic integration tests and
        development resets.
        """

        # --------------------------------------------------------------
        # Members must be deleted first because they reference clusters
        # through a foreign key.
        # --------------------------------------------------------------

        delete_members = text(
            f"""
            DELETE FROM {SCHEMA}.face_cluster_members
            """
        )

        self.db.execute(
            delete_members
        )

        # --------------------------------------------------------------
        # Now clusters can safely be deleted.
        # --------------------------------------------------------------

        delete_clusters = text(
            f"""
            DELETE FROM {SCHEMA}.face_clusters
            """
        )

        self.db.execute(
            delete_clusters
        )

        # --------------------------------------------------------------
        # Commit the reset.
        # --------------------------------------------------------------

        self.db.commit()

    # ==================================================================
    # TRANSACTION
    # ==================================================================

    def commit(self) -> None:
        """
        Commit the current transaction.
        """

        self.db.commit()

    def rollback(self) -> None:
        """
        Roll back the current transaction.
        """

        self.db.rollback()