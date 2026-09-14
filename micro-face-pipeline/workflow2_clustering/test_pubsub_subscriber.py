from __future__ import annotations

import json
from dataclasses import dataclass
from uuid import uuid4

from app.messaging.pubsub_subscriber import (
    Workflow2PubSubSubscriber,
)
from app.models.face_extraction_completed import (
    FaceExtractionCompletedEvent,
)


# ======================================================================
# TEST PUB/SUB MESSAGE
# ======================================================================

@dataclass
class FakePubSubMessage:
    """
    Minimal Pub/Sub message double.

    The real subscriber only requires:
        message.data
        message.ack()
        message.nack()
    """

    data: bytes

    ack_count: int = 0
    nack_count: int = 0

    def ack(self) -> None:
        self.ack_count += 1

    def nack(self) -> None:
        self.nack_count += 1


# ======================================================================
# TEST PROCESSOR
# ======================================================================

class FakeSuccessfulProcessor:
    """
    Processor double for transport-level subscriber tests.
    """

    def __init__(self) -> None:
        self.calls: list[
            FaceExtractionCompletedEvent
        ] = []

    def process(
        self,
        event: FaceExtractionCompletedEvent,
    ):
        self.calls.append(event)

        return FakeProcessResult(
            status="COMPLETED",
        )


class FakeFailedProcessor:
    """
    Processor double that deliberately fails.
    """

    def __init__(self) -> None:
        self.calls: list[
            FaceExtractionCompletedEvent
        ] = []

    def process(
        self,
        event: FaceExtractionCompletedEvent,
    ):
        self.calls.append(event)

        raise RuntimeError(
            "Intentional processor failure for test."
        )


class FakeUnsuccessfulProcessor:
    """
    Processor double that returns an unsuccessful result.
    """

    def __init__(self) -> None:
        self.calls: list[
            FaceExtractionCompletedEvent
        ] = []

    def process(
        self,
        event: FaceExtractionCompletedEvent,
    ):
        self.calls.append(event)

        return FakeProcessResult(
            status="FAILED",
        )


@dataclass(frozen=True)
class FakeProcessResult:
    status: str

    @property
    def success(self) -> bool:
        return self.status == "COMPLETED"


# ======================================================================
# EVENT FACTORY
# ======================================================================

def make_valid_payload(
    *,
    event_type: str = "FACE_EXTRACTION_COMPLETED",
    face_count: int = 5,
) -> dict:
    """
    Create a valid Workflow 1 -> Workflow 2 completion event payload.
    """

    return {
        "event_version": "1.0",
        "event_type": event_type,
        "event_id": str(uuid4()),
        "upload_id": 1,
        "image_id": 1,
        "bucket": "photo-micro-upload-test",
        "blob_name": (
            "uploads/"
            "009b6713-10e5-4a77-8e6e-ab61bd977e4b.jpg"
        ),
        "face_count": face_count,
    }


def make_message(
    payload: dict,
) -> FakePubSubMessage:
    return FakePubSubMessage(
        data=json.dumps(
            payload
        ).encode("utf-8")
    )


# ======================================================================
# ASSERTION HELPERS
# ======================================================================

def fail(message: str) -> None:
    raise AssertionError(message)


def assert_equal(
    actual,
    expected,
    message: str,
) -> None:
    if actual != expected:
        fail(
            f"{message} "
            f"Expected={expected!r}, "
            f"Actual={actual!r}"
        )


def assert_true(
    value: bool,
    message: str,
) -> None:
    if not value:
        fail(message)


# ======================================================================
# TEST 1 — VALID MESSAGE
# ======================================================================

def test_valid_message_is_processed_and_acked() -> None:
    print()
    print(
        "[1] Testing valid FACE_EXTRACTION_COMPLETED message..."
    )

    processor = FakeSuccessfulProcessor()

    subscriber = Workflow2PubSubSubscriber(
        processor=processor
    )

    payload = make_valid_payload()

    message = make_message(
        payload
    )

    subscriber.callback(
        message
    )

    assert_equal(
        len(processor.calls),
        1,
        "Processor call count mismatch.",
    )

    event = processor.calls[0]

    assert_equal(
        event.event_type,
        "FACE_EXTRACTION_COMPLETED",
        "Event type mismatch.",
    )

    assert_equal(
        event.image_id,
        1,
        "Image ID mismatch.",
    )

    assert_equal(
        event.face_count,
        5,
        "Face count mismatch.",
    )

    assert_equal(
        message.ack_count,
        1,
        "Valid message must be ACKed exactly once.",
    )

    assert_equal(
        message.nack_count,
        0,
        "Valid message must not be NACKed.",
    )

    print(
        "Valid message → processor → ACK: PASS"
    )


# ======================================================================
# TEST 2 — INVALID JSON
# ======================================================================

def test_invalid_json_is_nacked() -> None:
    print()
    print(
        "[2] Testing malformed JSON..."
    )

    processor = FakeSuccessfulProcessor()

    subscriber = Workflow2PubSubSubscriber(
        processor=processor
    )

    message = FakePubSubMessage(
        data=b"{this-is-not-valid-json"
    )

    subscriber.callback(
        message
    )

    assert_equal(
        len(processor.calls),
        0,
        "Processor must not run for invalid JSON.",
    )

    assert_equal(
        message.ack_count,
        0,
        "Invalid JSON must not be ACKed.",
    )

    assert_equal(
        message.nack_count,
        1,
        "Invalid JSON must be NACKed exactly once.",
    )

    print(
        "Malformed JSON → NACK: PASS"
    )


# ======================================================================
# TEST 3 — INVALID EVENT SCHEMA
# ======================================================================

def test_invalid_event_schema_is_nacked() -> None:
    print()
    print(
        "[3] Testing invalid event schema..."
    )

    processor = FakeSuccessfulProcessor()

    subscriber = Workflow2PubSubSubscriber(
        processor=processor
    )

    payload = make_valid_payload()

    # Remove required field.
    del payload["image_id"]

    message = make_message(
        payload
    )

    subscriber.callback(
        message
    )

    assert_equal(
        len(processor.calls),
        0,
        "Processor must not run for invalid event schema.",
    )

    assert_equal(
        message.ack_count,
        0,
        "Invalid event must not be ACKed.",
    )

    assert_equal(
        message.nack_count,
        1,
        "Invalid event must be NACKed exactly once.",
    )

    print(
        "Invalid event schema → NACK: PASS"
    )


# ======================================================================
# TEST 4 — WRONG EVENT TYPE
# ======================================================================

def test_wrong_event_type_is_nacked() -> None:
    print()
    print(
        "[4] Testing wrong event_type..."
    )

    processor = FakeSuccessfulProcessor()

    subscriber = Workflow2PubSubSubscriber(
        processor=processor
    )

    payload = make_valid_payload(
        event_type="SOMETHING_ELSE"
    )

    message = make_message(
        payload
    )

    subscriber.callback(
        message
    )

    assert_equal(
        len(processor.calls),
        0,
        "Processor must not run for wrong event type.",
    )

    assert_equal(
        message.ack_count,
        0,
        "Wrong event type must not be ACKed.",
    )

    assert_equal(
        message.nack_count,
        1,
        "Wrong event type must be NACKed exactly once.",
    )

    print(
        "Wrong event type → NACK: PASS"
    )


# ======================================================================
# TEST 5 — PROCESSOR FAILURE
# ======================================================================

def test_processor_failure_is_nacked() -> None:
    print()
    print(
        "[5] Testing processor failure..."
    )

    processor = FakeFailedProcessor()

    subscriber = Workflow2PubSubSubscriber(
        processor=processor
    )

    payload = make_valid_payload()

    message = make_message(
        payload
    )

    subscriber.callback(
        message
    )

    assert_equal(
        len(processor.calls),
        1,
        "Processor should receive the valid event.",
    )

    assert_equal(
        message.ack_count,
        0,
        "Failed processing must not be ACKed.",
    )

    assert_equal(
        message.nack_count,
        1,
        "Failed processing must be NACKed exactly once.",
    )

    print(
        "Processor failure → NACK: PASS"
    )


# ======================================================================
# TEST 6 — UNSUCCESSFUL PROCESSOR RESULT
# ======================================================================

def test_unsuccessful_processor_result_is_nacked() -> None:
    print()
    print(
        "[6] Testing unsuccessful processor result..."
    )

    processor = FakeUnsuccessfulProcessor()

    subscriber = Workflow2PubSubSubscriber(
        processor=processor
    )

    payload = make_valid_payload()

    message = make_message(
        payload
    )

    subscriber.callback(
        message
    )

    assert_equal(
        len(processor.calls),
        1,
        "Processor should receive the valid event.",
    )

    assert_equal(
        message.ack_count,
        0,
        "Unsuccessful processing must not be ACKed.",
    )

    assert_equal(
        message.nack_count,
        1,
        "Unsuccessful processing must be NACKed exactly once.",
    )

    print(
        "Unsuccessful processor result → NACK: PASS"
    )


# ======================================================================
# TEST 7 — DIRECT EVENT DECODING
# ======================================================================

def test_event_decoding() -> None:
    print()
    print(
        "[7] Testing event decoding..."
    )

    processor = FakeSuccessfulProcessor()

    subscriber = Workflow2PubSubSubscriber(
        processor=processor
    )

    payload = make_valid_payload()

    data = json.dumps(
        payload
    ).encode("utf-8")

    event = subscriber._decode_event(
        data
    )

    assert_true(
        isinstance(
            event,
            FaceExtractionCompletedEvent,
        ),
        "Decoded object is not FaceExtractionCompletedEvent.",
    )

    assert_equal(
        event.event_type,
        "FACE_EXTRACTION_COMPLETED",
        "Decoded event type mismatch.",
    )

    assert_equal(
        event.upload_id,
        1,
        "Decoded upload ID mismatch.",
    )

    assert_equal(
        event.image_id,
        1,
        "Decoded image ID mismatch.",
    )

    assert_equal(
        event.face_count,
        5,
        "Decoded face count mismatch.",
    )

    print(
        "Pub/Sub bytes → domain event: PASS"
    )


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    print(
        "=" * 70
    )
    print(
        "WORKFLOW 2 PUB/SUB SUBSCRIBER TEST"
    )
    print(
        "=" * 70
    )

    test_valid_message_is_processed_and_acked()

    test_invalid_json_is_nacked()

    test_invalid_event_schema_is_nacked()

    test_wrong_event_type_is_nacked()

    test_processor_failure_is_nacked()

    test_unsuccessful_processor_result_is_nacked()

    test_event_decoding()

    print()
    print(
        "=" * 70
    )
    print(
        "WORKFLOW 2 PUB/SUB SUBSCRIBER: PASS"
    )
    print(
        "=" * 70
    )

    print()
    print(
        "Verified:"
    )
    print(
        "  Valid event → processor → ACK       : PASS"
    )
    print(
        "  Invalid JSON → NACK                  : PASS"
    )
    print(
        "  Invalid event schema → NACK          : PASS"
    )
    print(
        "  Wrong event type → NACK              : PASS"
    )
    print(
        "  Processor exception → NACK           : PASS"
    )
    print(
        "  Unsuccessful processor → NACK        : PASS"
    )
    print(
        "  Event decoding                        : PASS"
    )


if __name__ == "__main__":
    main()