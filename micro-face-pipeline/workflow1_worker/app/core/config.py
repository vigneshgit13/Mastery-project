from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parent.parent.parent

DOWNLOAD_DIR = BASE_DIR / "temp"


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

    Password is intentionally read from the environment
    rather than being stored in source code.
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