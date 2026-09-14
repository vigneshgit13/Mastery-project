from pathlib import Path
import os

from dotenv import load_dotenv


# ============================================================
# BASE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DOWNLOAD_DIR = BASE_DIR / "temp"


# ============================================================
# ENVIRONMENT
# ============================================================

ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)


# ============================================================
# GCP
# ============================================================

PROJECT_ID = os.getenv(
    "PROJECT_ID",
    "test-face-clustering",
)

PUBSUB_SUBSCRIPTION = os.getenv(
    "PUBSUB_SUBSCRIPTION",
    "photo-upload-events-sub",
)

PUBSUB_COMPLETION_TOPIC = os.getenv(
    "PUBSUB_COMPLETION_TOPIC",
    "wf1-batch-completed",
)

# ============================================================
# PostgreSQL
# ============================================================

POSTGRES_HOST = os.getenv(
    "POSTGRES_HOST",
    "localhost",
)

POSTGRES_PORT = int(
    os.getenv(
        "POSTGRES_PORT",
        "5432",
    )
)

POSTGRES_DATABASE = os.getenv(
    "POSTGRES_DATABASE",
    "face_pipeline",
)

POSTGRES_USER = os.getenv(
    "POSTGRES_USER",
    "postgres",
)

POSTGRES_PASSWORD = os.getenv(
    "POSTGRES_PASSWORD",
)


def get_postgres_connection_string() -> str:
    """
    Build the PostgreSQL connection string.

    PostgreSQL credentials are loaded from the project's
    .env file/environment.
    """

    if not POSTGRES_PASSWORD:
        raise RuntimeError(
            "POSTGRES_PASSWORD environment variable is not set."
        )

    return (
        f"host={POSTGRES_HOST} "
        f"port={POSTGRES_PORT} "
        f"dbname={POSTGRES_DATABASE} "
        f"user={POSTGRES_USER} "
        f"password={POSTGRES_PASSWORD}"
    )


# ============================================================
# Redis
# ============================================================

REDIS_HOST = os.getenv(
    "REDIS_HOST",
    "localhost",
)

REDIS_PORT = int(
    os.getenv(
        "REDIS_PORT",
        "6379",
    )
)

REDIS_DB = int(
    os.getenv(
        "REDIS_DB",
        "0",
    )
)

REDIS_PASSWORD = os.getenv(
    "REDIS_PASSWORD",
)