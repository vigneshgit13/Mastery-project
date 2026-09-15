from unittest.mock import Mock

from app.messaging.pubsub_subscriber import Workflow2PubSubSubscriber


class FailingProcessor:
    def process(self, event):
        raise RuntimeError("TEST FAILURE: simulated Workflow 2 failure")


def main():
    subscriber = Workflow2PubSubSubscriber(
        processor=FailingProcessor()
    )

    message = Mock()

    message.data = (
        b'{"event_version":"1.0",'
        b'"event_type":"FACE_EXTRACTION_COMPLETED",'
        b'"event_id":"60e77165-8f38-495e-8046-a4d32e54122b",'
        b'"upload_id":6,'
        b'"image_id":6,'
        b'"bucket":"photo-micro-upload-test",'
        b'"blob_name":"uploads/78a6cb95-1fc6-42fd-9edd-a2a4836415cf.jpg",'
        b'"face_count":21}'
    )

    subscriber.callback(message)

    message.ack.assert_not_called()
    message.nack.assert_called_once()

    print()
    print("=" * 70)
    print("NACK TEST PASSED")
    print("=" * 70)
    print("ACK calls :", message.ack.call_count)
    print("NACK calls:", message.nack.call_count)


if __name__ == "__main__":
    main()