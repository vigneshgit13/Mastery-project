from __future__ import annotations

import uuid

from app.db.postgres import SessionLocal
from app.db.repository import Workflow1OutboxRepository


def main() -> None:

    print("=" * 70)
    print("WORKFLOW 1 OUTBOX REPOSITORY TEST")
    print("=" * 70)

    event_id = str(uuid.uuid4())

    with SessionLocal() as db:

        repo = Workflow1OutboxRepository(db)

        print("[1] Creating outbox event...")

        event = repo.create(
            event_id=event_id,
            event_type="FACE_EXTRACTION_COMPLETED",
            aggregate_type="upload",
            aggregate_id="1",
            payload='{"upload_id": 1, "image_count": 1}',
        )

        assert event.id is not None
        assert event.event_id == event_id
        assert event.status == "PENDING"
        assert event.attempt_count == 0

        db.commit()

        print("Create: PASS")

    with SessionLocal() as db:

        repo = Workflow1OutboxRepository(db)

        print("[2] Reading pending events...")

        pending = repo.get_pending()

        matching = [
            e for e in pending
            if e.event_id == event_id
        ]

        assert len(matching) == 1

        event = matching[0]

        assert event.status == "PENDING"
        assert event.attempt_count == 0

        print("Pending lookup: PASS")

        print("[3] Marking attempt...")

        repo.mark_attempt(event)

        assert event.attempt_count == 1

        db.commit()

        print("Attempt tracking: PASS")

    with SessionLocal() as db:

        repo = Workflow1OutboxRepository(db)

        pending = repo.get_pending()

        event = next(
            e for e in pending
            if e.event_id == event_id
        )

        print("[4] Marking published...")

        repo.mark_published(event)

        assert event.status == "PUBLISHED"
        assert event.published_at is not None
        assert event.last_error is None

        db.commit()

        print("Published state: PASS")

    print("=" * 70)
    print("WORKFLOW 1 OUTBOX REPOSITORY: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()