from __future__ import annotations

from app.db.postgres import SessionLocal
from app.db.repository import Workflow1OutboxRepository
from app.publisher.workflow1_outbox_publisher import (
    Workflow1OutboxPublisher,
)


print("=" * 70)
print("WORKFLOW 1 OUTBOX PUBLISHER TEST")
print("=" * 70)


# ================================================================
# 1. Inspect pending events
# ================================================================

db = SessionLocal()

try:

    repository = Workflow1OutboxRepository(db)

    pending = repository.get_pending()

    print(
        f"[1] Pending outbox events: {len(pending)}"
    )

    for event in pending:

        print(
            f"    id={event.id} "
            f"event_id={event.event_id} "
            f"status={event.status} "
            f"attempts={event.attempt_count}"
        )

finally:

    db.close()


# ================================================================
# 2. Create publisher
# ================================================================

print()
print("[2] Creating Workflow1OutboxPublisher...")

publisher = Workflow1OutboxPublisher()

print("Publisher: OK")


# ================================================================
# 3. Publish pending events
# ================================================================

print()
print("[3] Publishing pending events...")

count = publisher.publish_pending()

print(
    f"Published successfully: {count}"
)


# ================================================================
# 4. Verify PostgreSQL state
# ================================================================

print()
print("[4] Verifying PostgreSQL outbox state...")

db = SessionLocal()

try:

    repository = Workflow1OutboxRepository(db)

    pending = repository.get_pending()

    print(
        f"Remaining PENDING events: {len(pending)}"
    )

finally:

    db.close()


publisher.close()


print()
print("=" * 70)
print("WORKFLOW 1 OUTBOX PUBLISHER TEST COMPLETE")
print("=" * 70)