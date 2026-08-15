import logging



from app.publishers.pubsub_publisher import publisher
from app.services.upload_event_service import upload_event_service


logging.basicConfig(level=logging.INFO)


def gcs_trigger(event, context):
    """
    Cloud Storage Trigger (Gen2)

    Receives a GCS object finalize event,
    converts it into our UploadEvent,
    and publishes it to Pub/Sub.
    """

    logging.info("========== Cloud Function Triggered ==========")

    logging.info(f"Event ID       : {context.event_id}")
    logging.info(f"Event Type     : {context.event_type}")
    logging.info(f"Bucket         : {event.get('bucket')}")
    logging.info(f"Object         : {event.get('name')}")
    logging.info(f"Content Type   : {event.get('contentType')}")
    logging.info(f"Size           : {event.get('size')}")

    # ------------------------------------------------------------------
    # Show which credentials Cloud Function is using
    # ------------------------------------------------------------------
   


    # ------------------------------------------------------------------
    # Convert GCS Event -> UploadEvent
    # ------------------------------------------------------------------
    upload_event = upload_event_service.create_event(event)

    logging.info("UploadEvent created successfully.")

    logging.info(upload_event.model_dump_json(indent=2))

    # ------------------------------------------------------------------
    # Publish to Pub/Sub
    # ------------------------------------------------------------------
    try:

        message_id = publisher.publish(upload_event)

        logging.info(
            f"Successfully published message: {message_id}"
        )

    except Exception:

        logging.exception("Failed to publish message to Pub/Sub.")

        raise

    logging.info("========== Function Completed ==========")