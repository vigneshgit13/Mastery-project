"""
FAISS/HNSW vector-store service for Workflow 1.

Responsibilities
----------------
- Maintain a 512-D face-embedding index.
- Use FAISS IndexHNSWFlat for approximate nearest-neighbor search.
- Normalize embeddings before insertion/search.
- Use inner-product similarity, which is cosine similarity for
  normalized vectors.
- Maintain a mapping between FAISS internal integer IDs and
  application-level vector IDs.
- Persist the FAISS index and mapping to disk.
- Provide idempotent insertion.
- Provide thread-safe access.

Architecture
------------

FacePipeline
     |
     | 512-D ArcFace embedding
     v
FaissService
     |
     +---- FAISS HNSW index
     |          |
     |          +---- actual embedding vector
     |
     +---- vector ID mapping
                |
                v
        PostgreSQL face_embeddings

PostgreSQL remains the authoritative metadata/state database.
The actual embedding vector is stored in FAISS/HNSW.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import faiss
import numpy as np


logger = logging.getLogger(__name__)


class FaissService:
    """
    Thread-safe FAISS HNSW service.

    Default configuration is designed for the existing ArcFace
    recognizer in Workflow 1:

        embedding dimension = 512
        HNSW M              = 32
        efConstruction      = 200
        efSearch            = 64
        normalized vectors  = True
        metric              = inner product

    With normalized vectors, inner product is equivalent to
    cosine similarity.
    """

    def __init__(
        self,
        index_path: str | Path = "data/hnsw/index.bin",
        metadata_path: str | Path = "data/hnsw/metadata.json",
        dimension: int = 512,
        m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
        normalize: bool = True,
    ) -> None:
        self.dimension = dimension
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.normalize = normalize

        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)

        # Thread-safety for worker access.
        self._lock = threading.RLock()

        # Mapping:
        #
        # FAISS internal integer ID
        #             |
        #             v
        # application vector_id
        #
        # Example:
        #
        # 0 -> "face:101"
        # 1 -> "face:102"
        self._vector_ids: dict[int, str] = {}

        # Additional application metadata.
        #
        # Example:
        #
        # "face:101" -> {
        #     "face_id": 101,
        #     "image_id": 1
        # }
        self._metadata: dict[str, dict[str, Any]] = {}

        # Next application-level FAISS position.
        self._next_vector_number = 0

        # Create a new index or load the existing one.
        self.index = self._create_or_load_index()

    # ================================================================
    # INDEX CREATION
    # ================================================================

    def _create_index(self) -> faiss.Index:
        """
        Create a new FAISS HNSW index.
        """

        metric = (
            faiss.METRIC_INNER_PRODUCT
            if self.normalize
            else faiss.METRIC_L2
        )

        index = faiss.IndexHNSWFlat(
            self.dimension,
            self.m,
            metric,
        )

        index.hnsw.efConstruction = (
            self.ef_construction
        )

        index.hnsw.efSearch = (
            self.ef_search
        )

        logger.info(
            "Created FAISS HNSW index: "
            "dimension=%d, M=%d, efConstruction=%d, "
            "efSearch=%d, normalize=%s",
            self.dimension,
            self.m,
            self.ef_construction,
            self.ef_search,
            self.normalize,
        )

        return index

    # ================================================================
    # INDEX LOAD / INITIALIZATION
    # ================================================================

    def _create_or_load_index(self) -> faiss.Index:
        """
        Create a new FAISS index or load an existing persisted index.
        """

        self.index_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.metadata_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ------------------------------------------------------------
        # Existing index
        # ------------------------------------------------------------

        if self.index_path.exists():
            logger.info(
                "Loading FAISS index from: %s",
                self.index_path,
            )

            index = faiss.read_index(
                str(self.index_path)
            )

            # Validate embedding dimension.
            if index.d != self.dimension:
                raise ValueError(
                    "FAISS dimension mismatch: "
                    f"index={index.d}, "
                    f"configured={self.dimension}"
                )

            # Restore HNSW search parameter.
            if hasattr(index, "hnsw"):
                index.hnsw.efSearch = (
                    self.ef_search
                )

            # IMPORTANT:
            #
            # Pass the loaded index explicitly.
            #
            # We cannot use self.index here because
            # self.index has not yet been assigned during
            # __init__.
            self._load_metadata(index)

            logger.info(
                "FAISS index loaded successfully: "
                "vectors=%d",
                index.ntotal,
            )

            return index

        # ------------------------------------------------------------
        # New index
        # ------------------------------------------------------------

        logger.info(
            "No existing FAISS index found. "
            "Creating a new HNSW index."
        )

        return self._create_index()

    # ================================================================
    # METADATA LOAD
    # ================================================================

    def _load_metadata(
        self,
        index: faiss.Index,
    ) -> None:
        """
        Load vector-ID metadata associated with a FAISS index.

        IMPORTANT:
        The FAISS index is passed explicitly because during object
        initialization self.index does not exist yet.
        """

        # ------------------------------------------------------------
        # Metadata file doesn't exist
        # ------------------------------------------------------------

        if not self.metadata_path.exists():

            # If vectors already exist, we cannot safely determine
            # which application vector ID belongs to which vector.
            if index.ntotal:
                raise RuntimeError(
                    "FAISS index exists without metadata mapping."
                )

            self._vector_ids = {}
            self._metadata = {}
            self._next_vector_number = 0

            return

        # ------------------------------------------------------------
        # Read metadata
        # ------------------------------------------------------------

        logger.info(
            "Loading FAISS metadata from: %s",
            self.metadata_path,
        )

        with self.metadata_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            payload = json.load(f)

        # ------------------------------------------------------------
        # Restore FAISS ID -> vector_id mapping
        # ------------------------------------------------------------

        self._vector_ids = {
            int(key): str(value)
            for key, value in payload.get(
                "vector_ids",
                {},
            ).items()
        }

        # ------------------------------------------------------------
        # Restore application metadata
        # ------------------------------------------------------------

        self._metadata = payload.get(
            "metadata",
            {},
        )

        # ------------------------------------------------------------
        # Restore next vector number
        # ------------------------------------------------------------

        default_next_number = (
            max(
                self._vector_ids,
                default=-1,
            )
            + 1
        )

        self._next_vector_number = int(
            payload.get(
                "next_vector_number",
                default_next_number,
            )
        )

        # ------------------------------------------------------------
        # Validate index <-> metadata consistency
        # ------------------------------------------------------------

        if index.ntotal != len(
            self._vector_ids
        ):
            raise RuntimeError(
                "FAISS index/vector mapping mismatch: "
                f"index.ntotal={index.ntotal}, "
                f"mapped={len(self._vector_ids)}"
            )

        # ------------------------------------------------------------
        # Validate that metadata has corresponding vector IDs
        # ------------------------------------------------------------

        for vector_id in self._vector_ids.values():

            if vector_id not in self._metadata:
                logger.warning(
                    "Vector ID '%s' has no application metadata.",
                    vector_id,
                )

        logger.info(
            "FAISS metadata loaded successfully: "
            "mapped_vectors=%d, next_vector_number=%d",
            len(self._vector_ids),
            self._next_vector_number,
        )

    # ================================================================
    # SAVE
    # ================================================================

    def save(self) -> None:
        """
        Persist the FAISS index and application metadata.

        Files:
            index.bin
            metadata.json
        """

        with self._lock:

            self.index_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            self.metadata_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            # --------------------------------------------------------
            # Save FAISS index
            # --------------------------------------------------------

            faiss.write_index(
                self.index,
                str(self.index_path),
            )

            # --------------------------------------------------------
            # Save mapping metadata
            # --------------------------------------------------------

            payload = {
                "dimension": self.dimension,
                "metric": (
                    "inner_product"
                    if self.normalize
                    else "l2"
                ),
                "vector_ids": {
                    str(key): value
                    for key, value
                    in self._vector_ids.items()
                },
                "metadata": self._metadata,
                "next_vector_number": (
                    self._next_vector_number
                ),
            }

            with self.metadata_path.open(
                "w",
                encoding="utf-8",
            ) as f:

                json.dump(
                    payload,
                    f,
                    indent=2,
                )

            logger.info(
                "FAISS index saved successfully: "
                "vectors=%d, path=%s",
                self.index.ntotal,
                self.index_path,
            )

    # ================================================================
    # EMBEDDING PREPARATION
    # ================================================================

    def _prepare_embedding(
        self,
        embedding: Any,
    ) -> np.ndarray:
        """
        Validate and prepare one embedding for FAISS.

        Accepted input:
            shape (512,)
            shape (1, 512)

        Returned shape:
            (1, 512)
        """

        vector = np.asarray(
            embedding,
            dtype=np.float32,
        )

        # ------------------------------------------------------------
        # Convert (512,) -> (1,512)
        # ------------------------------------------------------------

        if vector.ndim == 1:
            vector = vector.reshape(
                1,
                -1,
            )

        # ------------------------------------------------------------
        # Validate shape
        # ------------------------------------------------------------

        if (
            vector.ndim != 2
            or vector.shape[0] != 1
        ):
            raise ValueError(
                "Embedding must contain exactly "
                "one vector. "
                f"Received shape={vector.shape}"
            )

        # ------------------------------------------------------------
        # Validate dimension
        # ------------------------------------------------------------

        if vector.shape[1] != self.dimension:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"received={vector.shape[1]}, "
                f"expected={self.dimension}"
            )

        # ------------------------------------------------------------
        # Validate numerical values
        # ------------------------------------------------------------

        if not np.isfinite(vector).all():
            raise ValueError(
                "Embedding contains NaN or infinite values."
            )

        # ------------------------------------------------------------
        # Normalize
        # ------------------------------------------------------------

        if self.normalize:

            norm = np.linalg.norm(
                vector,
                axis=1,
                keepdims=True,
            )

            if float(norm[0, 0]) <= 0.0:
                raise ValueError(
                    "Cannot normalize a zero-norm embedding."
                )

            vector = (
                vector / norm
            )

        return np.ascontiguousarray(
            vector,
            dtype=np.float32,
        )

    # ================================================================
    # ADD ONE EMBEDDING
    # ================================================================

    def add(
        self,
        embedding: Any,
        vector_id: str,
        metadata: dict[str, Any] | None = None,
        persist: bool = True,
    ) -> dict[str, Any]:
        """
        Add one face embedding.

        The operation is idempotent by vector_id.

        Example:

            service.add(
                embedding,
                vector_id="face:101",
                metadata={
                    "face_id": 101,
                    "image_id": 1,
                },
            )
        """

        if not vector_id:
            raise ValueError(
                "vector_id must not be empty."
            )

        with self._lock:

            # --------------------------------------------------------
            # Idempotency check
            # --------------------------------------------------------

            for (
                faiss_id,
                existing_vector_id,
            ) in self._vector_ids.items():

                if existing_vector_id == vector_id:

                    logger.info(
                        "Vector already exists: "
                        "vector_id=%s, faiss_id=%d",
                        vector_id,
                        faiss_id,
                    )

                    return {
                        "vector_id": vector_id,
                        "faiss_id": faiss_id,
                        "added": False,
                    }

            # --------------------------------------------------------
            # Prepare vector
            # --------------------------------------------------------

            vector = self._prepare_embedding(
                embedding
            )

            # --------------------------------------------------------
            # Assign internal FAISS ID
            # --------------------------------------------------------

            faiss_id = (
                self._next_vector_number
            )

            # --------------------------------------------------------
            # Add to HNSW
            # --------------------------------------------------------

            self.index.add(vector)

            # --------------------------------------------------------
            # Store mapping
            # --------------------------------------------------------

            self._vector_ids[
                faiss_id
            ] = vector_id

            self._metadata[
                vector_id
            ] = metadata or {}

            self._next_vector_number += 1

            # --------------------------------------------------------
            # Persist
            # --------------------------------------------------------

            if persist:
                self.save()

            logger.info(
                "Embedding added: "
                "vector_id=%s, faiss_id=%d",
                vector_id,
                faiss_id,
            )

            return {
                "vector_id": vector_id,
                "faiss_id": faiss_id,
                "added": True,
            }

    # ================================================================
    # ADD MANY
    # ================================================================

    def add_many(
        self,
        embeddings: list[Any],
        vector_ids: list[str],
        metadata: list[
            dict[str, Any] | None
        ] | None = None,
        persist: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Add multiple embeddings.

        Persistence is performed once at the end rather than after
        every vector.
        """

        if len(embeddings) != len(
            vector_ids
        ):
            raise ValueError(
                "embeddings and vector_ids "
                "must have the same length."
            )

        if (
            metadata is not None
            and len(metadata)
            != len(embeddings)
        ):
            raise ValueError(
                "metadata must have the same "
                "length as embeddings."
            )

        results: list[
            dict[str, Any]
        ] = []

        for (
            index,
            (
                embedding,
                vector_id,
            ),
        ) in enumerate(
            zip(
                embeddings,
                vector_ids,
            )
        ):

            item_metadata = (
                metadata[index]
                if metadata is not None
                else None
            )

            result = self.add(
                embedding=embedding,
                vector_id=vector_id,
                metadata=item_metadata,
                persist=False,
            )

            results.append(result)

        if persist:
            self.save()

        return results

    # ================================================================
    # SEARCH
    # ================================================================

    def search(
        self,
        embedding: Any,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search nearest neighbors.

        When normalize=True:

            score = cosine similarity

        because normalized vectors are searched using
        inner-product similarity.
        """

        if k <= 0:
            raise ValueError(
                "k must be greater than zero."
            )

        with self._lock:

            # --------------------------------------------------------
            # Empty index
            # --------------------------------------------------------

            if self.index.ntotal == 0:
                return []

            # --------------------------------------------------------
            # Prepare query
            # --------------------------------------------------------

            vector = self._prepare_embedding(
                embedding
            )

            # --------------------------------------------------------
            # Limit k to available vectors
            # --------------------------------------------------------

            actual_k = min(
                k,
                self.index.ntotal,
            )

            # --------------------------------------------------------
            # Search
            # --------------------------------------------------------

            scores, ids = self.index.search(
                vector,
                actual_k,
            )

            results: list[
                dict[str, Any]
            ] = []

            # --------------------------------------------------------
            # Convert FAISS IDs to application IDs
            # --------------------------------------------------------

            for (
                score,
                faiss_id,
            ) in zip(
                scores[0],
                ids[0],
            ):

                faiss_id = int(
                    faiss_id
                )

                # FAISS can return -1 for an unavailable result.
                if faiss_id < 0:
                    continue

                vector_id = (
                    self._vector_ids.get(
                        faiss_id
                    )
                )

                if vector_id is None:

                    logger.warning(
                        "FAISS returned an "
                        "unmapped internal ID: %s",
                        faiss_id,
                    )

                    continue

                results.append(
                    {
                        "vector_id": vector_id,
                        "faiss_id": faiss_id,
                        "score": float(score),
                        "metadata": (
                            self._metadata.get(
                                vector_id,
                                {},
                            )
                        ),
                    }
                )

            return results

    # ================================================================
    # CONTAINS
    # ================================================================

    def contains(
        self,
        vector_id: str,
    ) -> bool:
        """
        Check whether an application vector ID exists.
        """

        with self._lock:
            return (
                vector_id
                in self._metadata
            )

    # ================================================================
    # METADATA
    # ================================================================

    def get_metadata(
        self,
        vector_id: str,
    ) -> dict[str, Any] | None:
        """
        Retrieve metadata associated with a vector ID.
        """

        with self._lock:
            return self._metadata.get(
                vector_id
            )

    # ================================================================
    # COUNT
    # ================================================================

    @property
    def count(self) -> int:
        """
        Number of vectors currently stored in FAISS.
        """

        return int(
            self.index.ntotal
        )

    # ================================================================
    # CLOSE
    # ================================================================

    def close(self) -> None:
        """
        Persist current state.

        FAISS does not require an explicit close operation, but
        explicitly saving here gives the worker a clean lifecycle
        operation.
        """

        self.save()