from __future__ import annotations

import logging
from pathlib import Path

from app.clustering.clustering_processor import (
    ClusteringProcessor,
)
from app.clustering.clustering_service import (
    ClusteringService,
)
from app.clustering.hnsw_store import (
    HNSWVectorStore,
)
from app.clustering.vector_bootstrap import (
    Workflow1VectorReader,
)
from app.core.config import (
    WORKFLOW1_DIR,
)
from app.db.clustering_repository import (
    ClusteringRepository,
)
from app.db.postgres import (
    SessionLocal,
)
from app.messaging.subscriber_runtime import (
    Workflow2SubscriberRuntime,
)


logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s "
        "%(levelname)s "
        "%(name)s "
        "%(message)s"
    ),
)

logger = logging.getLogger(__name__)


DIMENSION = 512

SIMILARITY_THRESHOLD = 0.5

SEARCH_K = 5


def build_processor(
    db,
) -> ClusteringProcessor:
    """
    Construct the complete Workflow 2 business stack.

    PostgreSQL
        authoritative metadata/state

    Workflow 1 HNSW
        read-only vector source

    Workflow 2 HNSW
        active clustering/search layer
    """

    repository = ClusteringRepository(
        db=db,
    )

    # --------------------------------------------------------------
    # Workflow 1 source HNSW
    # --------------------------------------------------------------

    workflow1_hnsw_dir = (
        WORKFLOW1_DIR
        / "data"
        / "hnsw"
    )

    workflow1_index = (
        workflow1_hnsw_dir
        / "index.bin"
    )

    workflow1_metadata = (
        workflow1_hnsw_dir
        / "metadata.json"
    )

    logger.info(
        "Workflow 1 HNSW index: %s",
        workflow1_index,
    )

    logger.info(
        "Workflow 1 HNSW metadata: %s",
        workflow1_metadata,
    )

    source_store = Workflow1VectorReader(
        index_path=workflow1_index,
        metadata_path=workflow1_metadata,
        dimension=DIMENSION,
    )

    # --------------------------------------------------------------
    # Workflow 2 active HNSW
    # --------------------------------------------------------------

    workflow2_hnsw_dir = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "hnsw"
    )

    workflow2_index = (
        workflow2_hnsw_dir
        / "index.bin"
    )

    workflow2_metadata = (
        workflow2_hnsw_dir
        / "metadata.json"
    )

    logger.info(
        "Workflow 2 HNSW index: %s",
        workflow2_index,
    )

    logger.info(
        "Workflow 2 HNSW metadata: %s",
        workflow2_metadata,
    )

    active_store = HNSWVectorStore(
        index_path=workflow2_index,
        metadata_path=workflow2_metadata,
        dimension=DIMENSION,
        M=32,
        ef_construction=200,
        ef_search=64,
    )

    # --------------------------------------------------------------
    # Clustering service
    # --------------------------------------------------------------

    clustering_service = ClusteringService(
        repository=repository,
        source_store=source_store,
        active_store=active_store,
        similarity_threshold=SIMILARITY_THRESHOLD,
        search_k=SEARCH_K,
    )

    # --------------------------------------------------------------
    # Clustering processor
    # --------------------------------------------------------------

    processor = ClusteringProcessor(
        repository=repository,
        clustering_service=clustering_service,
    )

    return processor


def main() -> None:
    logger.info(
        "============================================================"
    )
    logger.info(
        "WORKFLOW 2 CLUSTERING WORKER"
    )
    logger.info(
        "============================================================"
    )

    db = SessionLocal()

    try:

        processor = build_processor(
            db=db,
        )

        runtime = Workflow2SubscriberRuntime(
            processor=processor,
        )

        runtime.run()

    finally:

        db.close()

        logger.info(
            "Workflow 2 PostgreSQL session closed."
        )


if __name__ == "__main__":
    main()