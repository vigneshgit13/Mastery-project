from pydantic import BaseModel
from google.cloud import pubsub_v1
import google.auth
from google.auth.transport.requests import Request

from app.core.config import (
    PROJECT_ID,
    PUBSUB_TOPIC,
)


class PubSubPublisher:

    def __init__(self):

        credentials, project = google.auth.default()
        credentials.refresh(Request())

        print("=" * 80)
        print("GOOGLE AUTH")
        print("Credential type :", type(credentials))
        print("Project         :", project)
        print(
            "SA              :",
            getattr(credentials, "service_account_email", "UNKNOWN"),
        )
        print("=" * 80)

        self.publisher = pubsub_v1.PublisherClient(
            credentials=credentials
        )

        self.topic_path = self.publisher.topic_path(
            PROJECT_ID,
            PUBSUB_TOPIC,
        )

    def publish(self, event: BaseModel):
        print("=" * 80)
        print("Publishing to:", self.topic_path)
                
        credentials, project = google.auth.default()
        credentials.refresh(Request())
                
        print("Credential:", type(credentials))
        print(
              "Email:",
               getattr(credentials, "service_account_email", "UNKNOWN"),
                    )
        print("=" * 80)
                
        message = event.model_dump_json().encode("utf-8")
                
        try:
            future = self.publisher.publish(
            self.topic_path,
            message,
        )
            
            print("Publish request sent")
                        
            message_id = future.result(timeout=30)
                        
            print("Message ID:", message_id)
                        
            return message_id
            
        except Exception as e:
             print("PUBLISH FAILED")
             print(type(e))
             print(e)
             raise
        
               


publisher = PubSubPublisher()