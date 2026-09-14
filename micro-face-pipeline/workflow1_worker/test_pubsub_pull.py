from google.cloud import pubsub_v1
from app.subscriber.pubsub_subscriber import (
    PROJECT_ID,
    PUBSUB_SUBSCRIPTION,
)
import time


def callback(message):
    print("=" * 80, flush=True)
    print("DIRECT PUB/SUB MESSAGE RECEIVED", flush=True)
    print("MESSAGE ID:", message.message_id, flush=True)
    print("DATA:", message.data.decode("utf-8"), flush=True)
    print("=" * 80, flush=True)

    message.ack()

    print("MESSAGE ACKED", flush=True)


subscriber = pubsub_v1.SubscriberClient()

subscription_path = subscriber.subscription_path(
    PROJECT_ID,
    PUBSUB_SUBSCRIPTION,
)

print("PROJECT_ID:", PROJECT_ID, flush=True)
print("SUBSCRIPTION:", subscription_path, flush=True)
print("STARTING STREAMING PULL...", flush=True)

streaming_future = subscriber.subscribe(
    subscription_path,
    callback=callback,
)

try:
    streaming_future.result(
        timeout=30
    )

except Exception as exc:
    print(
        "STREAMING RESULT:",
        repr(exc),
        flush=True,
    )

finally:
    streaming_future.cancel()
    subscriber.close()

print("TEST FINISHED", flush=True)