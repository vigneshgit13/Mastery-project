from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from app.clustering.hnsw_store import HNSWVectorStore
from app.db.clustering_repository import ClusteringRepository


class Workflow1VectorReader:
    """
    Read-only reader for Workflow 1's persisted HNSW/FAISS store.

    Production behavior:

    - Workflow 2 may start before Workflow 1 has created its vector files.
    - Missing Workflow 1 files are treated as an empty source store.
    - The reader refreshes automatically when vectors become available.
    - Workflow 1 remains the owner of the source index.
    """

    def __init__(
        self,
        index_path: str | Path,
        metadata_path: str | Path,
        dimension: int = 512,
    ) -> None:

        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.dimension = int(dimension)

        self.index = None
        self.metadata: dict = {}
        self.vector_to_faiss: dict[str, int] = {}

        self._loaded_index_mtime_ns: int | None = None
        self._loaded_metadata_mtime_ns: int | None = None

        # Attempt initial load.
        #
        # IMPORTANT:
        # Missing files are valid when Workflow 1 has not yet processed
        # its first image.
        self.refresh(force=True)

    # ==================================================================
    # STORE STATE
    # ==================================================================

    def is_available(self) -> bool:
        """
        Return True when both Workflow 1 vector files exist.
        """

        return (
            self.index_path.exists()
            and self.metadata_path.exists()
        )

    def _clear(self) -> None:
        """
        Reset this reader to an empty state.
        """

        self.index = None
        self.metadata = {}
        self.vector_to_faiss = {}

        self._loaded_index_mtime_ns = None
        self._loaded_metadata_mtime_ns = None

    # ==================================================================
    # REFRESH / LOAD
    # ==================================================================

    def refresh(
        self,
        *,
        force: bool = False,
    ) -> bool:
        """
        Refresh Workflow 1 vector state.

        Returns:
            True  -> store is currently available and loaded
            False -> Workflow 1 store does not yet exist

        The reader reloads automatically when either index.bin or
        metadata.json changes.
        """

        # --------------------------------------------------------------
        # Workflow 1 has not processed anything yet.
        # This is a valid state.
        # --------------------------------------------------------------

        if not self.is_available():

            self._clear()

            return False

        index_mtime_ns = (
            self.index_path.stat().st_mtime_ns
        )

        metadata_mtime_ns = (
            self.metadata_path.stat().st_mtime_ns
        )

        needs_reload = (
            force
            or self.index is None
            or self._loaded_index_mtime_ns
            != index_mtime_ns
            or self._loaded_metadata_mtime_ns
            != metadata_mtime_ns
        )

        if not needs_reload:
            return True

        # --------------------------------------------------------------
        # Load FAISS/HNSW index.
        # --------------------------------------------------------------

        index = faiss.read_index(
            str(self.index_path)
        )

        # --------------------------------------------------------------
        # Load metadata.
        # --------------------------------------------------------------

        with self.metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            metadata = json.load(file)

        # --------------------------------------------------------------
        # Validate loaded data before replacing active state.
        # --------------------------------------------------------------

        self._validate_metadata(
            metadata=metadata,
        )

        vector_to_faiss = (
            self._build_mappings(
                metadata=metadata,
                index=index,
            )
        )

        # --------------------------------------------------------------
        # Atomically replace reader state.
        # --------------------------------------------------------------

        self.index = index
        self.metadata = metadata
        self.vector_to_faiss = vector_to_faiss

        self._loaded_index_mtime_ns = (
            index_mtime_ns
        )

        self._loaded_metadata_mtime_ns = (
            metadata_mtime_ns
        )

        return True

    # ==================================================================
    # VALIDATION
    # ==================================================================

    def _validate_metadata(
        self,
        *,
        metadata: dict,
    ) -> None:

        if not isinstance(
            metadata,
            dict,
        ):

            raise RuntimeError(
                "Workflow 1 metadata must be a JSON object."
            )

        metadata_dimension = metadata.get(
            "dimension"
        )

        if metadata_dimension is None:

            raise RuntimeError(
                "Workflow 1 metadata is missing "
                "'dimension'."
            )

        if int(metadata_dimension) != self.dimension:

            raise RuntimeError(
                "Workflow 1 metadata dimension mismatch: "
                f"expected={self.dimension}, "
                f"actual={metadata_dimension}"
            )

        metric = metadata.get(
            "metric"
        )

        if metric != "inner_product":

            raise RuntimeError(
                "Unexpected Workflow 1 vector metric: "
                f"{metric!r}"
            )

        vector_ids = metadata.get(
            "vector_ids"
        )

        if not isinstance(
            vector_ids,
            dict,
        ):

            raise RuntimeError(
                "Workflow 1 metadata 'vector_ids' "
                "must be an object."
            )

    # ==================================================================
    # MAPPINGS
    # ==================================================================

    def _build_mappings(
        self,
        *,
        metadata: dict,
        index,
    ) -> dict[str, int]:

        vector_ids = metadata[
            "vector_ids"
        ]

        vector_to_faiss: dict[str, int] = {}

        for (
            faiss_id_string,
            vector_id,
        ) in vector_ids.items():

            try:

                faiss_id = int(
                    faiss_id_string
                )

            except (
                TypeError,
                ValueError,
            ) as exc:

                raise RuntimeError(
                    "Invalid FAISS ID in Workflow 1 "
                    f"metadata: {faiss_id_string!r}"
                ) from exc

            vector_id = str(
                vector_id
            )

            if vector_id in vector_to_faiss:

                raise RuntimeError(
                    "Duplicate vector_id in Workflow 1 "
                    f"metadata: {vector_id}"
                )

            vector_to_faiss[
                vector_id
            ] = faiss_id

        # --------------------------------------------------------------
        # Validate metadata against index.
        # --------------------------------------------------------------

        if len(
            vector_to_faiss
        ) != index.ntotal:

            raise RuntimeError(
                "Workflow 1 vector metadata/index "
                "count mismatch: "
                f"metadata={len(vector_to_faiss)}, "
                f"index={index.ntotal}"
            )

        return vector_to_faiss

    # ==================================================================
    # VECTOR RETRIEVAL
    # ==================================================================

    def get_embedding(
        self,
        vector_id: str,
    ) -> np.ndarray | None:
        """
        Retrieve one embedding from Workflow 1.

        Automatically refreshes before lookup so Workflow 2 can start
        before Workflow 1 has created its first index.
        """

        # --------------------------------------------------------------
        # Refresh source store.
        # --------------------------------------------------------------

        available = self.refresh()

        if not available:

            return None

        # Defensive guard.
        if self.index is None:

            return None

        faiss_id = (
            self.vector_to_faiss.get(
                vector_id
            )
        )

        if faiss_id is None:

            return None

        faiss_id = int(
            faiss_id
        )

        if (
            faiss_id < 0
            or faiss_id >= self.index.ntotal
        ):

            raise RuntimeError(
                "Invalid Workflow 1 FAISS ID: "
                f"vector_id={vector_id}, "
                f"faiss_id={faiss_id}, "
                f"ntotal={self.index.ntotal}"
            )

        # --------------------------------------------------------------
        # Reconstruct vector.
        # --------------------------------------------------------------

        vector = self.index.reconstruct(
            faiss_id
        )

        vector = np.asarray(
            vector,
            dtype=np.float32,
        ).reshape(-1)

        # --------------------------------------------------------------
        # Dimension validation.
        # --------------------------------------------------------------

        if vector.shape != (
            self.dimension,
        ):

            raise RuntimeError(
                "Workflow 1 vector dimension mismatch: "
                f"vector_id={vector_id}, "
                f"expected={self.dimension}, "
                f"actual={vector.shape}"
            )

        # --------------------------------------------------------------
        # Numerical validation.
        # --------------------------------------------------------------

        if not np.isfinite(
            vector
        ).all():

            raise RuntimeError(
                "Workflow 1 vector contains "
                f"NaN/Inf: {vector_id}"
            )

        norm = float(
            np.linalg.norm(
                vector
            )
        )

        if norm <= 0.0:

            raise RuntimeError(
                "Workflow 1 vector has zero norm: "
                f"{vector_id}"
            )

        if not np.isclose(
            norm,
            1.0,
            atol=1e-4,
        ):

            raise RuntimeError(
                "Workflow 1 vector is not normalized: "
                f"vector_id={vector_id}, "
                f"norm={norm}"
            )

        return vector

    # ==================================================================
    # VECTOR IDS
    # ==================================================================

    def get_vector_ids(
        self,
    ) -> list[str]:

        available = self.refresh()

        if not available:

            return []

        return sorted(
            self.vector_to_faiss.keys(),
            key=lambda vector_id: (
                self.vector_to_faiss[
                    vector_id
                ]
            ),
        )

    # ==================================================================
    # COUNT
    # ==================================================================

    def count(self) -> int:

        available = self.refresh()

        if not available:

            return 0

        if self.index is None:

            return 0

        return int(
            self.index.ntotal
        )
# ======================================================================
# WORKFLOW 1 -> WORKFLOW 2 BOOTSTRAP
# ======================================================================

class Workflow1ToWorkflow2Bootstrap:
    """
    Bootstrap Workflow 2's independent HNSW index using the real
    vectors persisted by Workflow 1.

    PostgreSQL:
        authoritative metadata

    Workflow 1 HNSW:
        source of actual vectors

    Workflow 2 HNSW:
        independent clustering/search replica
    """

    def __init__(
        self,
        repository: ClusteringRepository,
        source_store: Workflow1VectorReader,
        target_store: HNSWVectorStore,
    ) -> None:

        self.repository = repository
        self.source_store = source_store
        self.target_store = target_store

    # ==================================================================
    # BOOTSTRAP
    # ==================================================================

    def run(
        self,
    ) -> dict[str, int]:

        records = (
            self.repository
            .get_all_face_embeddings()
        )

        source_records = 0
        added_count = 0
        skipped_count = 0

        for record in records:

            source_records += 1

            vector_id = (
                record.vector_id
            )

            # ----------------------------------------------------------
            # Retrieve actual vector from Workflow 1.
            # ----------------------------------------------------------

            vector = (
                self.source_store
                .get_embedding(
                    vector_id
                )
            )

            if vector is None:
                raise RuntimeError(
                    "Vector registered in PostgreSQL "
                    "was not found in Workflow 1 HNSW: "
                    f"face_id={record.face_id}, "
                    f"vector_id={vector_id}"
                )

            # ----------------------------------------------------------
            # Validate dimension against PostgreSQL.
            # ----------------------------------------------------------

            if vector.shape != (
                record.embedding_dimension,
            ):

                raise RuntimeError(
                    "Embedding dimension mismatch: "
                    f"face_id={record.face_id}, "
                    f"vector_id={vector_id}, "
                    f"postgres={record.embedding_dimension}, "
                    f"actual={vector.shape[0]}"
                )

            # ----------------------------------------------------------
            # No cluster assignment yet.
            #
            # ClusteringService owns cluster assignment.
            # ----------------------------------------------------------

            added = self.target_store.add(
                embedding=vector,
                vector_id=vector_id,
                cluster_id=None,
            )

            if added:
                added_count += 1
            else:
                skipped_count += 1

        return {
            "source_records": source_records,
            "added": added_count,
            "skipped_existing": skipped_count,
            "target_vectors": self.target_store.count(),
        }