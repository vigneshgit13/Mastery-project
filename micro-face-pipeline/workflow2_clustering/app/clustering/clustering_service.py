from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.clustering.hnsw_store import HNSWVectorStore
from app.clustering.vector_bootstrap import Workflow1VectorReader
from app.db.clustering_repository import (
    ClusteringRepository,
    FaceEmbeddingRecord,
)


@dataclass(frozen=True)
class ClusterAssignment:
    """
    Result of clustering one face.
    """

    face_id: int
    vector_id: str
    cluster_id: int
    similarity: float
    created_new_cluster: bool


class ClusteringService:
    """
    Workflow 2 identity clustering service.

    Responsibilities
    ----------------
    1. Read authoritative embedding metadata from PostgreSQL.
    2. Retrieve the actual 512-D embedding from Workflow 1 HNSW.
    3. Search the active Workflow 2 HNSW.
    4. Compare nearest-neighbor similarity against a threshold.
    5. Assign the face to an existing cluster OR create a new cluster.
    6. Persist the cluster/member relationship in PostgreSQL.
    7. Insert the vector into Workflow 2 HNSW with its cluster ID.

    Identity matching is based on ArcFace embedding similarity only.

    Gender and age are NOT used for identity matching.
    """

    DEFAULT_SIMILARITY_THRESHOLD = 0.50
    DEFAULT_SEARCH_K = 5

    def __init__(
        self,
        repository: ClusteringRepository,
        source_store: Workflow1VectorReader,
        active_store: HNSWVectorStore,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        search_k: int = DEFAULT_SEARCH_K,
    ) -> None:

        self.repository = repository
        self.source_store = source_store
        self.active_store = active_store

        self.similarity_threshold = float(
            similarity_threshold
        )

        self.search_k = int(search_k)

        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError(
                "similarity_threshold must be between "
                "0.0 and 1.0"
            )

        if self.search_k <= 0:
            raise ValueError(
                "search_k must be greater than zero"
            )

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    def process_all(
        self,
    ) -> list[ClusterAssignment]:
        """
        Process all embeddings currently registered in PostgreSQL.

        Faces are processed in face_id order because
        get_all_face_embeddings() returns deterministic ordering.

        This gives us deterministic initial clustering behavior.
        """

        records = (
            self.repository
            .get_all_face_embeddings()
        )

        results: list[
            ClusterAssignment
        ] = []

        for record in records:

            result = self.process_face(
                record
            )

            results.append(
                result
            )

        return results

    # ==================================================================
    # SINGLE FACE
    # ==================================================================

    def process_face(
        self,
        record: FaceEmbeddingRecord,
    ) -> ClusterAssignment:
        """
        Cluster one face.

        Idempotency:
            If PostgreSQL already contains a cluster membership for
            this face, return that existing assignment rather than
            assigning it again.
        """

        # --------------------------------------------------------------
        # 1. Check existing PostgreSQL membership.
        # --------------------------------------------------------------

        existing_cluster = (
            self.repository
            .get_cluster_for_face(
                record.face_id
            )
        )

        if existing_cluster is not None:

            return ClusterAssignment(
                face_id=record.face_id,
                vector_id=record.vector_id,
                cluster_id=existing_cluster,
                similarity=1.0,
                created_new_cluster=False,
            )
       
        # --------------------------------------------------------------
        # 2. Retrieve actual embedding from Workflow 1.
        # --------------------------------------------------------------

        embedding = (
            self.source_store
            .get_embedding(
                record.vector_id
            )
        )

        if embedding is None:

            raise RuntimeError(
                "Embedding referenced by PostgreSQL "
                "was not found in Workflow 1 HNSW: "
                f"face_id={record.face_id}, "
                f"vector_id={record.vector_id}"
            )

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        ).reshape(-1)

        # --------------------------------------------------------------
        # 3. Validate against PostgreSQL metadata.
        # --------------------------------------------------------------

        if embedding.shape != (
            record.embedding_dimension,
        ):

            raise RuntimeError(
                "Embedding dimension mismatch: "
                f"face_id={record.face_id}, "
                f"vector_id={record.vector_id}, "
                f"postgres={record.embedding_dimension}, "
                f"actual={embedding.shape[0]}"
            )

        if not np.isfinite(
            embedding
        ).all():

            raise RuntimeError(
                "Embedding contains NaN/Inf: "
                f"vector_id={record.vector_id}"
            )

        norm = float(
            np.linalg.norm(
                embedding
            )
        )

        if norm <= 0.0:

            raise RuntimeError(
                "Embedding has zero norm: "
                f"vector_id={record.vector_id}"
            )

        # --------------------------------------------------------------
        # 4. Search active Workflow 2 HNSW.
        # --------------------------------------------------------------

        candidates = (
            self.active_store.search(
                embedding,
                k=self.search_k,
            )
        )

        # --------------------------------------------------------------
        # 5. Decide cluster.
        # --------------------------------------------------------------

        if not candidates:

            # First vector in the active index.
            cluster_id = (
                self._create_new_cluster(
                    record=record,
                    similarity=1.0,
                )
            )

            self._insert_into_active_store(
                record=record,
                embedding=embedding,
                cluster_id=cluster_id,
            )

            return ClusterAssignment(
                face_id=record.face_id,
                vector_id=record.vector_id,
                cluster_id=cluster_id,
                similarity=1.0,
                created_new_cluster=True,
            )

        # --------------------------------------------------------------
        # Nearest candidate is first because HNSW search is ordered
        # by descending inner-product similarity.
        # --------------------------------------------------------------

        nearest = candidates[0]

        nearest_similarity = float(
            nearest["similarity"]
        )

        nearest_cluster_id = int(
            nearest["cluster_id"]
        )

        # --------------------------------------------------------------
        # Debug/trace information.
        # --------------------------------------------------------------

        print(
            "[CLUSTER MATCH] "
            f"face_id={record.face_id} "
            f"vector_id={record.vector_id} "
            f"nearest_vector={nearest['vector_id']} "
            f"cluster_id={nearest_cluster_id} "
            f"similarity={nearest_similarity:.6f} "
            f"threshold={self.similarity_threshold:.6f}"
        )

        # --------------------------------------------------------------
        # 6A. Existing cluster.
        # --------------------------------------------------------------

        if (
            nearest_similarity
            >= self.similarity_threshold
        ):

            self._persist_membership(
                record=record,
                cluster_id=nearest_cluster_id,
                similarity=nearest_similarity,
            )

            self._insert_into_active_store(
                record=record,
                embedding=embedding,
                cluster_id=nearest_cluster_id,
            )

            return ClusterAssignment(
                face_id=record.face_id,
                vector_id=record.vector_id,
                cluster_id=nearest_cluster_id,
                similarity=nearest_similarity,
                created_new_cluster=False,
            )

        # --------------------------------------------------------------
        # 6B. No sufficiently similar cluster.
        # --------------------------------------------------------------

        cluster_id = (
            self._create_new_cluster(
                record=record,
                similarity=1.0,
            )
        )

        self._insert_into_active_store(
            record=record,
            embedding=embedding,
            cluster_id=cluster_id,
        )

        return ClusterAssignment(
            face_id=record.face_id,
            vector_id=record.vector_id,
            cluster_id=cluster_id,
            similarity=1.0,
            created_new_cluster=True,
        )

    # ==================================================================
    # CREATE CLUSTER
    # ==================================================================

    def _create_new_cluster(
        self,
        *,
        record: FaceEmbeddingRecord,
        similarity: float,
    ) -> int:
        """
        Create a PostgreSQL cluster and add its first member.
        """

        try:

            cluster_id = (
                self.repository
                .create_cluster()
            )

            self.repository.add_cluster_member(
                cluster_id=cluster_id,
                face_id=record.face_id,
                similarity=float(
                    similarity
                ),
            )

            self.repository.commit()

            print(
                "[CLUSTER CREATED] "
                f"cluster_id={cluster_id} "
                f"face_id={record.face_id} "
                f"vector_id={record.vector_id}"
            )

            return cluster_id

        except Exception:

            self.repository.rollback()

            raise

    # ==================================================================
    # EXISTING CLUSTER MEMBERSHIP
    # ==================================================================

    def _persist_membership(
        self,
        *,
        record: FaceEmbeddingRecord,
        cluster_id: int,
        similarity: float,
    ) -> None:
        """
        Add a face to an existing PostgreSQL cluster.
        """

        try:

            self.repository.add_cluster_member(
                cluster_id=cluster_id,
                face_id=record.face_id,
                similarity=float(
                    similarity
                ),
            )

            self.repository.commit()

            print(
                "[CLUSTER ASSIGNED] "
                f"cluster_id={cluster_id} "
                f"face_id={record.face_id} "
                f"similarity={similarity:.6f}"
            )

        except Exception:

            self.repository.rollback()

            raise

    # ==================================================================
    # ACTIVE HNSW INSERTION
    # ==================================================================

    def _insert_into_active_store(
        self,
        *,
        record: FaceEmbeddingRecord,
        embedding: np.ndarray,
        cluster_id: int,
    ) -> None:
        """
        Insert the vector into Workflow 2 HNSW.

        persist=False is used because the clustering service controls
        the operation sequence. The HNSW store is persisted after the
        insertion succeeds.
        """

        try:

            added = (
                self.active_store.add(
                    embedding=embedding,
                    vector_id=record.vector_id,
                    cluster_id=cluster_id,
                    persist=False,
                )
            )

            if not added:

                print(
                    "[HNSW] "
                    f"Already present: "
                    f"{record.vector_id}"
                )

            self.active_store.save()

        except Exception:

            raise