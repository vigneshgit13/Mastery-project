from __future__ import annotations

import logging
import sys
from pathlib import Path

from app.ai.pipeline_factory import create_face_pipeline
from app.models.upload_event import UploadEvent
from app.services.processing_service import ProcessingService
from datetime import datetime, timezone


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def main() -> None:
    print("=" * 70)
    print("WORKFLOW 1 PROCESSING SERVICE E2E TEST")
    print("=" * 70)

    # ------------------------------------------------------------
    # 1. Create the real AI pipeline
    # ------------------------------------------------------------

    print("\n[1] Creating FacePipeline...")

    pipeline = create_face_pipeline()

    print("FacePipeline: OK")

    # ------------------------------------------------------------
    # 2. Create ProcessingService
    # ------------------------------------------------------------

    print("\n[2] Creating ProcessingService...")

    service = ProcessingService(
        pipeline=pipeline,
    )

    print("ProcessingService: OK")

    # ------------------------------------------------------------
    # 3. Build a test UploadEvent
    #
    # IMPORTANT:
    # This event must point to an actual image already available
    # in your configured GCS bucket.
    # ------------------------------------------------------------

    event = UploadEvent(
        event_id="22222222-2222-4222-8222-222222222222",
        event_version="1.0",
        event_type="google.cloud.storage.object.v1.finalized",
        project_id="test-face-clustering",
        bucket="photo-micro-upload-test",
        blob_name="uploads/842224b3-783d-48d2-9489-74bfe263453f.jpg",
        original_filename="photo-micro-upload-test/uploads/842224b3-783d-48d2-9489-74bfe263453f.jpg",
        content_type="image/jpeg",
        file_size=0,
        uploaded_at=datetime.now(timezone.utc),
        created_by="e2e-test",
    )

    print("\n[3] UploadEvent created")
    print("Event ID :", event.event_id)
    print("Bucket   :", event.bucket)
    print("Blob     :", event.blob_name)

    # ------------------------------------------------------------
    # 4. Process
    # ------------------------------------------------------------

    print("\n[4] Running ProcessingService...")
    print("This will execute:")
    print("  GCS")
    print("   -> PostgreSQL")
    print("   -> FacePipeline")
    print("   -> PostgreSQL faces")
    print("   -> FAISS/HNSW")
    print("   -> PostgreSQL embeddings")
    print("   -> Redis")
    print()

    result = service.process_upload(event)

    # ------------------------------------------------------------
    # 5. Print result
    # ------------------------------------------------------------

    print("\n[5] Processing result")
    print("-" * 70)

    for key, value in result.items():
        print(f"{key}: {value}")

    print("-" * 70)

    # ------------------------------------------------------------
    # 6. Basic validation
    # ------------------------------------------------------------

    assert result["status"] in {
        "COMPLETED",
        "ALREADY_COMPLETED",
    }

    print("\nSTATUS CHECK: PASS")

    if result["status"] == "COMPLETED":
        assert result["face_count"] >= 0

        print(
            f"Face count: {result['face_count']}"
        )

        assert "faces" in result

        print("Face result structure: PASS")

    print("\n" + "=" * 70)
    print("PROCESSING SERVICE E2E TEST: PASS")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n" + "=" * 70)
        print("PROCESSING SERVICE E2E TEST: FAILED")
        print("=" * 70)
        raise