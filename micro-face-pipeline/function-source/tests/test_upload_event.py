from app.services.upload_event_service import (
    upload_event_service,
)


sample_event = {
    "bucket": "photo-micro-upload-test",
    "name": "uploads/test.jpg",
    "contentType": "image/jpeg",
    "size": "123456",
}

event = upload_event_service.create_event(sample_event)

print(event.model_dump_json(indent=2))