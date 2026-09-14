from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import faiss
import numpy as np


class HNSWVectorStore:
    """
    Workflow 2 HNSW/FAISS vector store.

    Responsibilities
    ----------------
    - Store 512-D face embeddings.
    - Use FAISS IndexHNSWFlat.
    - Use inner-product similarity.
    - Normalize embeddings before insertion/search.
    - Maintain application vector ID -> FAISS ID mapping.
    - Maintain FAISS ID -> application vector ID mapping.
    - Maintain vector ID -> cluster ID mapping.
    - Persist the FAISS index.
    - Persist the mapping metadata.
    - Support idempotent insertion.
    - Support similarity search.
    - Validate index/metadata consistency during reload.

    Architecture
    ------------
    PostgreSQL
        |
        +-- face_embeddings
        |
        +-- face_clusters
        |
        +-- face_cluster_members
        |
        +----------------------+
                               |
                               v
                        Workflow 2 HNSW
                               |
                               +-- 512-D vectors
                               +-- vector IDs
                               +-- cluster mapping

    PostgreSQL remains the authoritative metadata store.
    This class is only responsible for the active vector-search layer
    and its local vector-to-application mappings.
    """

    def __init__(
        self,
        index_path: str | Path = "data/hnsw/index.bin",
        metadata_path: str | Path = "data/hnsw/metadata.json",
        dimension: int = 512,
        M: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
    ) -> None:

        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)

        self.dimension = int(dimension)
        self.M = int(M)
        self.ef_construction = int(ef_construction)
        self.ef_search = int(ef_search)

        self._lock = threading.RLock()

        # --------------------------------------------------------------
        # Application vector ID -> FAISS internal integer ID
        #
        # Example:
        #   "face:101" -> 0
        # --------------------------------------------------------------
        self.vector_to_faiss: dict[str, int] = {}

        # --------------------------------------------------------------
        # FAISS internal integer ID -> application vector ID
        #
        # Example:
        #   0 -> "face:101"
        # --------------------------------------------------------------
        self.faiss_to_vector: dict[int, str] = {}

        # --------------------------------------------------------------
        # Application vector ID -> cluster ID
        #
        # Example:
        #   "face:101" -> 1
        # --------------------------------------------------------------
        self.vector_to_cluster: dict[str, int] = {}

        # --------------------------------------------------------------
        # Create a new index or load an existing one.
        #
        # IMPORTANT:
        # _create_or_load_index() returns the index object.
        # Metadata validation receives that index explicitly.
        # This avoids accessing self.index before __init__ has assigned it.
        # --------------------------------------------------------------
        self.index = self._create_or_load_index()

    # ==================================================================
    # INDEX CREATION / LOADING
    # ==================================================================

    def _create_index(self) -> faiss.Index:
        """
        Create a new FAISS HNSW index.

        Workflow 2 uses:
            - 512-D vectors
            - IndexHNSWFlat
            - inner-product similarity

        Normalized vectors + inner product ~= cosine similarity.
        """

        index = faiss.IndexHNSWFlat(
            self.dimension,
            self.M,
            faiss.METRIC_INNER_PRODUCT,
        )

        index.hnsw.efConstruction = self.ef_construction
        index.hnsw.efSearch = self.ef_search

        return index

    def _create_or_load_index(self) -> faiss.Index:
        """
        Create a new index or load the persisted index.

        If an existing index is found, its metadata is also loaded and
        validated against the actual FAISS index.
        """

        self.index_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.metadata_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # --------------------------------------------------------------
        # Existing persisted index
        # --------------------------------------------------------------

        if self.index_path.exists():

            index = faiss.read_index(
                str(self.index_path)
            )

            # ----------------------------------------------------------
            # Validate dimensionality.
            # ----------------------------------------------------------

            if index.d != self.dimension:
                raise ValueError(
                    "HNSW dimension mismatch: "
                    f"index={index.d}, "
                    f"configured={self.dimension}"
                )

            # ----------------------------------------------------------
            # Restore search parameter.
            # ----------------------------------------------------------

            if hasattr(index, "hnsw"):
                index.hnsw.efSearch = self.ef_search

            # ----------------------------------------------------------
            # IMPORTANT FIX:
            #
            # Pass the local index object into _load_metadata().
            #
            # Previously _load_metadata() tried to access self.index
            # before self.index had been assigned by __init__().
            # ----------------------------------------------------------

            self._load_metadata(index)

            return index

        # --------------------------------------------------------------
        # No persisted index.
        # --------------------------------------------------------------

        return self._create_index()

    # ==================================================================
    # EMBEDDING PREPARATION
    # ==================================================================

    def _prepare_embedding(
        self,
        embedding: np.ndarray,
    ) -> np.ndarray:
        """
        Validate and normalize one 512-D embedding.

        Accepted input:
            shape (512,)

        Internally converted to:
            shape (1, 512)

        Returns:
            float32 normalized vector.
        """

        vector = np.asarray(
            embedding,
            dtype=np.float32,
        )

        # --------------------------------------------------------------
        # Convert:
        #
        # (512,)
        #
        # to:
        #
        # (1, 512)
        # --------------------------------------------------------------

        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        # --------------------------------------------------------------
        # Validate shape.
        # --------------------------------------------------------------

        if vector.shape != (
            1,
            self.dimension,
        ):
            raise ValueError(
                "Invalid embedding shape: "
                f"expected (1, {self.dimension}), "
                f"got {vector.shape}"
            )

        # --------------------------------------------------------------
        # Validate numerical values.
        # --------------------------------------------------------------

        if not np.isfinite(vector).all():
            raise ValueError(
                "Embedding contains NaN or infinite values"
            )

        # --------------------------------------------------------------
        # L2 normalization.
        # --------------------------------------------------------------

        norm = np.linalg.norm(vector)

        if norm <= 0.0:
            raise ValueError(
                "Embedding has zero norm"
            )

        vector = vector / norm

        return vector.astype(
            np.float32,
            copy=False,
        )

    # ==================================================================
    # ADD
    # ==================================================================

    def add(
        self,
        embedding: np.ndarray,
        vector_id: str,
        cluster_id: int,
        persist: bool = True,
    ) -> bool:
        """
        Add one face embedding.

        Parameters
        ----------
        embedding:
            512-D face embedding.

        vector_id:
            Application-level vector identifier.

            Example:
                "face:101"

        cluster_id:
            Workflow 2 cluster identifier.

        persist:
            If True, immediately persist the index and metadata.

        Returns
        -------
        True:
            Vector was newly inserted.

        False:
            Vector already existed with the same cluster assignment.

        Raises
        ------
        ValueError:
            If the vector already exists but is associated with a
            different cluster.
        """

        # --------------------------------------------------------------
        # Validate vector ID.
        # --------------------------------------------------------------

        if not vector_id:
            raise ValueError(
                "vector_id must not be empty"
            )

        # --------------------------------------------------------------
        # Validate cluster ID.
        # --------------------------------------------------------------

        if cluster_id is None:
            raise ValueError(
                "cluster_id must not be None"
            )

        cluster_id = int(cluster_id)

        with self._lock:

            # ----------------------------------------------------------
            # Idempotency check.
            # ----------------------------------------------------------

            if vector_id in self.vector_to_faiss:

                existing_cluster = (
                    self.vector_to_cluster.get(
                        vector_id
                    )
                )

                # ------------------------------------------------------
                # Same vector + same cluster:
                #
                # Treat as idempotent duplicate.
                # ------------------------------------------------------

                if existing_cluster == cluster_id:
                    return False

                # ------------------------------------------------------
                # Same vector + different cluster:
                #
                # This is a consistency violation.
                # Do not silently overwrite it.
                # ------------------------------------------------------

                raise ValueError(
                    "Vector already exists with a different "
                    "cluster assignment: "
                    f"vector_id={vector_id}, "
                    f"existing_cluster={existing_cluster}, "
                    f"new_cluster={cluster_id}"
                )

            # ----------------------------------------------------------
            # Prepare embedding.
            # ----------------------------------------------------------

            vector = self._prepare_embedding(
                embedding
            )

            # ----------------------------------------------------------
            # FAISS IndexHNSWFlat assigns sequential internal IDs.
            #
            # Since vectors are never removed in this micro-version:
            #
            #   current ntotal == next FAISS ID
            # ----------------------------------------------------------

            faiss_id = int(
                self.index.ntotal
            )

            # ----------------------------------------------------------
            # Add vector.
            # ----------------------------------------------------------

            self.index.add(vector)

            # ----------------------------------------------------------
            # Maintain mappings.
            # ----------------------------------------------------------

            self.vector_to_faiss[
                vector_id
            ] = faiss_id

            self.faiss_to_vector[
                faiss_id
            ] = vector_id

            self.vector_to_cluster[
                vector_id
            ] = cluster_id

            # ----------------------------------------------------------
            # Persist.
            # ----------------------------------------------------------

            if persist:
                self.save()

            return True

    # ==================================================================
    # SEARCH
    # ==================================================================

    def search(
        self,
        embedding: np.ndarray,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search nearest face embeddings.

        Parameters
        ----------
        embedding:
            Query 512-D embedding.

        k:
            Maximum number of nearest neighbors.

        Returns
        -------
        List of dictionaries:

        [
            {
                "vector_id": "face:101",
                "cluster_id": 1,
                "faiss_id": 0,
                "similarity": 0.93,
            },
            ...
        ]

        Similarity is the FAISS inner-product result.

        Because embeddings are normalized, this corresponds to
        cosine similarity.
        """

        with self._lock:

            # ----------------------------------------------------------
            # Empty index.
            # ----------------------------------------------------------

            if self.index.ntotal == 0:
                return []

            # ----------------------------------------------------------
            # Prepare query vector.
            # ----------------------------------------------------------

            vector = self._prepare_embedding(
                embedding
            )

            # ----------------------------------------------------------
            # Clamp k so FAISS never receives a value larger than the
            # number of stored vectors.
            # ----------------------------------------------------------

            k = max(
                1,
                min(
                    int(k),
                    int(self.index.ntotal),
                ),
            )

            # ----------------------------------------------------------
            # FAISS nearest-neighbor search.
            # ----------------------------------------------------------

            similarities, indices = (
                self.index.search(
                    vector,
                    k,
                )
            )

            results: list[dict[str, Any]] = []

            # ----------------------------------------------------------
            # Convert FAISS internal IDs into application-level
            # information.
            # ----------------------------------------------------------

            for similarity, faiss_id in zip(
                similarities[0],
                indices[0],
            ):
                faiss_id = int(
                    faiss_id
                )

                # ------------------------------------------------------
                # FAISS can return -1 for an invalid/unavailable result.
                # ------------------------------------------------------

                if faiss_id < 0:
                    continue

                vector_id = (
                    self.faiss_to_vector.get(
                        faiss_id
                    )
                )

                if vector_id is None:
                    raise RuntimeError(
                        "FAISS index contains an ID that is missing "
                        "from faiss_to_vector metadata: "
                        f"faiss_id={faiss_id}"
                    )

                cluster_id = (
                    self.vector_to_cluster.get(
                        vector_id
                    )
                )

                if cluster_id is None:
                    raise RuntimeError(
                        "Vector is missing cluster metadata: "
                        f"vector_id={vector_id}"
                    )

                results.append(
                    {
                        "vector_id": vector_id,
                        "cluster_id": cluster_id,
                        "faiss_id": faiss_id,
                        "similarity": float(
                            similarity
                        ),
                    }
                )

            return results

    # ==================================================================
    # LOOKUPS
    # ==================================================================

    def contains(
        self,
        vector_id: str,
    ) -> bool:
        """
        Return True if the application vector ID exists.
        """

        with self._lock:
            return (
                vector_id
                in self.vector_to_faiss
            )

    def get_faiss_id(
        self,
        vector_id: str,
    ) -> int | None:
        """
        Return FAISS internal ID for an application vector ID.
        """

        with self._lock:
            return self.vector_to_faiss.get(
                vector_id
            )

    def get_vector_id(
        self,
        faiss_id: int,
    ) -> str | None:
        """
        Return application vector ID for a FAISS internal ID.
        """

        with self._lock:
            return self.faiss_to_vector.get(
                int(faiss_id)
            )

    def get_cluster_id(
        self,
        vector_id: str,
    ) -> int | None:
        """
        Return cluster ID associated with a vector.
        """

        with self._lock:
            return self.vector_to_cluster.get(
                vector_id
            )
        # ==================================================================
    # EMBEDDING RETRIEVAL
    # ==================================================================

    def get_embedding(
        self,
        vector_id: str,
    ) -> np.ndarray | None:
        """
        Retrieve the stored normalized embedding for an application
        vector ID.

        Parameters
        ----------
        vector_id:
            Application-level vector identifier.

            Example:
                "face:1"

        Returns
        -------
        numpy.ndarray | None
            The stored 512-D float32 embedding with shape:

                (512,)

            Returns None if the vector ID does not exist.

        Notes
        -----
        The actual embedding is owned by the FAISS/HNSW index.

        PostgreSQL only stores metadata such as:

            face_id
            vector_id
            model_name
            model_version
            embedding_dimension
            vector_store
            normalized
            embedding_norm

        Therefore this method retrieves the actual vector directly
        from the HNSW/FAISS index.
        """

        with self._lock:

            # ----------------------------------------------------------
            # Resolve application vector ID to FAISS internal ID.
            # ----------------------------------------------------------

            faiss_id = self.vector_to_faiss.get(
                vector_id
            )

            if faiss_id is None:
                return None

            # ----------------------------------------------------------
            # Validate FAISS internal ID.
            # ----------------------------------------------------------

            faiss_id = int(faiss_id)

            if faiss_id < 0:
                raise ValueError(
                    "Invalid FAISS ID for vector: "
                    f"vector_id={vector_id}, "
                    f"faiss_id={faiss_id}"
                )

            if faiss_id >= int(
                self.index.ntotal
            ):
                raise ValueError(
                    "FAISS ID is outside the current index: "
                    f"vector_id={vector_id}, "
                    f"faiss_id={faiss_id}, "
                    f"ntotal={self.index.ntotal}"
                )

            # ----------------------------------------------------------
            # Reconstruct the stored vector.
            #
            # IndexHNSWFlat supports reconstruction because the actual
            # vectors are retained by the underlying flat storage.
            # ----------------------------------------------------------

            vector = self.index.reconstruct(
                faiss_id
            )

            # ----------------------------------------------------------
            # Convert to the public representation:
            #
            # shape = (512,)
            # dtype = float32
            # ----------------------------------------------------------

            vector = np.asarray(
                vector,
                dtype=np.float32,
            ).reshape(-1)

            # ----------------------------------------------------------
            # Validate dimension.
            # ----------------------------------------------------------

            if vector.shape != (
                self.dimension,
            ):
                raise RuntimeError(
                    "Stored embedding dimension mismatch: "
                    f"vector_id={vector_id}, "
                    f"expected={self.dimension}, "
                    f"got={vector.shape}"
                )

            # ----------------------------------------------------------
            # Validate numerical values.
            # ----------------------------------------------------------

            if not np.isfinite(vector).all():
                raise RuntimeError(
                    "Stored embedding contains NaN or infinite values: "
                    f"vector_id={vector_id}"
                )

            # ----------------------------------------------------------
            # Validate that the stored vector is normalized.
            #
            # Workflow 2 normalizes all vectors before insertion.
            # ----------------------------------------------------------

            norm = float(
                np.linalg.norm(vector)
            )

            if norm <= 0.0:
                raise RuntimeError(
                    "Stored embedding has zero norm: "
                    f"vector_id={vector_id}"
                )

            # ----------------------------------------------------------
            # We don't renormalize the retrieved vector here.
            #
            # Retrieval should return what is actually stored.
            # We only validate that it is approximately unit length.
            # ----------------------------------------------------------

            if not np.isclose(
                norm,
                1.0,
                atol=1e-4,
            ):
                raise RuntimeError(
                    "Stored embedding is not normalized: "
                    f"vector_id={vector_id}, "
                    f"norm={norm}"
                )

            return vector
    

    def count(self) -> int:
        """
        Return number of vectors stored in HNSW.
        """

        with self._lock:
            return int(
                self.index.ntotal
            )

    # ==================================================================
    # PERSISTENCE
    # ==================================================================

    def save(self) -> None:
        """
        Persist both:

            1. FAISS HNSW index
            2. Application metadata mappings

        These are separate files because FAISS stores the vector
        structure while our application metadata stores the mapping
        between FAISS IDs, face/vector IDs, and cluster IDs.
        """

        with self._lock:

            # ----------------------------------------------------------
            # Ensure directories exist.
            # ----------------------------------------------------------

            self.index_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            self.metadata_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            # ----------------------------------------------------------
            # Persist FAISS index.
            # ----------------------------------------------------------

            faiss.write_index(
                self.index,
                str(self.index_path),
            )

            # ----------------------------------------------------------
            # Prepare metadata.
            # ----------------------------------------------------------

            metadata = {
                "dimension": self.dimension,
                "M": self.M,
                "ef_construction": (
                    self.ef_construction
                ),
                "ef_search": self.ef_search,

                "vector_to_faiss": (
                    self.vector_to_faiss
                ),

                "faiss_to_vector": {
                    str(key): value
                    for key, value
                    in self.faiss_to_vector.items()
                },

                "vector_to_cluster": (
                    self.vector_to_cluster
                ),
            }

            # ----------------------------------------------------------
            # Persist metadata.
            # ----------------------------------------------------------

            with self.metadata_path.open(
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    metadata,
                    file,
                    indent=2,
                    sort_keys=True,
                )

    # ==================================================================
    # METADATA LOADING
    # ==================================================================

    def _load_metadata(
        self,
        index: faiss.Index,
    ) -> None:
        """
        Load application metadata associated with an existing FAISS
        index.

        The FAISS index is passed explicitly because during __init__
        self.index has not yet been assigned.
        """

        # --------------------------------------------------------------
        # Metadata must exist if index exists.
        # --------------------------------------------------------------

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                "HNSW metadata file does not exist: "
                f"{self.metadata_path}"
            )

        # --------------------------------------------------------------
        # Read JSON.
        # --------------------------------------------------------------

        with self.metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            metadata = json.load(
                file
            )

        # --------------------------------------------------------------
        # Validate dimension.
        # --------------------------------------------------------------

        stored_dimension = int(
            metadata["dimension"]
        )

        if stored_dimension != self.dimension:
            raise ValueError(
                "Metadata dimension mismatch: "
                f"stored={stored_dimension}, "
                f"configured={self.dimension}"
            )

        # --------------------------------------------------------------
        # Restore vector -> FAISS mapping.
        # --------------------------------------------------------------

        self.vector_to_faiss = {
            str(key): int(value)
            for key, value
            in metadata[
                "vector_to_faiss"
            ].items()
        }

        # --------------------------------------------------------------
        # Restore FAISS -> vector mapping.
        # --------------------------------------------------------------

        self.faiss_to_vector = {
            int(key): str(value)
            for key, value
            in metadata[
                "faiss_to_vector"
            ].items()
        }

        # --------------------------------------------------------------
        # Restore vector -> cluster mapping.
        # --------------------------------------------------------------

        self.vector_to_cluster = {
            str(key): int(value)
            for key, value
            in metadata[
                "vector_to_cluster"
            ].items()
        }

        # ==============================================================
        # CONSISTENCY VALIDATION
        # ==============================================================

        index_count = int(
            index.ntotal
        )

        vector_to_faiss_count = len(
            self.vector_to_faiss
        )

        faiss_to_vector_count = len(
            self.faiss_to_vector
        )

        vector_to_cluster_count = len(
            self.vector_to_cluster
        )

        # --------------------------------------------------------------
        # All three metadata mappings must contain exactly one entry
        # for every vector in the FAISS index.
        # --------------------------------------------------------------

        if (
            vector_to_faiss_count
            != index_count
        ):
            raise ValueError(
                "HNSW metadata/index count mismatch: "
                f"vector_to_faiss="
                f"{vector_to_faiss_count}, "
                f"index={index_count}"
            )

        if (
            faiss_to_vector_count
            != index_count
        ):
            raise ValueError(
                "HNSW reverse metadata/index count mismatch: "
                f"faiss_to_vector="
                f"{faiss_to_vector_count}, "
                f"index={index_count}"
            )

        if (
            vector_to_cluster_count
            != index_count
        ):
            raise ValueError(
                "HNSW cluster metadata/index count mismatch: "
                f"vector_to_cluster="
                f"{vector_to_cluster_count}, "
                f"index={index_count}"
            )

        # --------------------------------------------------------------
        # Validate mapping consistency:
        #
        # vector -> FAISS -> vector
        # --------------------------------------------------------------

        for (
            vector_id,
            faiss_id,
        ) in self.vector_to_faiss.items():

            reverse_vector_id = (
                self.faiss_to_vector.get(
                    faiss_id
                )
            )

            if reverse_vector_id != vector_id:
                raise ValueError(
                    "HNSW mapping inconsistency: "
                    f"vector_id={vector_id}, "
                    f"faiss_id={faiss_id}, "
                    f"reverse_vector_id="
                    f"{reverse_vector_id}"
                )

        # --------------------------------------------------------------
        # Validate that every vector has a cluster assignment.
        # --------------------------------------------------------------

        for vector_id in self.vector_to_faiss:

            if vector_id not in (
                self.vector_to_cluster
            ):
                raise ValueError(
                    "HNSW vector is missing "
                    "cluster assignment: "
                    f"vector_id={vector_id}"
                )