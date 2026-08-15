import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent

load_dotenv(BASE_DIR / ".env")

PROJECT_ID = os.getenv(
    "PROJECT_ID",
    "test-face-clustering",
)

PUBSUB_TOPIC = os.getenv(
    "PUBSUB_TOPIC",
    "photo-upload-events",
)