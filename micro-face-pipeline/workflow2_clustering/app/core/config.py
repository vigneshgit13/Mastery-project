from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

WORKFLOW1_DIR = (
    BASE_DIR.parent / "workflow1_worker"
)


# ============================================================
# LOAD WORKFLOW 2 ENVIRONMENT
# ============================================================

workflow2_env = BASE_DIR / ".env"

if workflow2_env.exists():
    load_dotenv(
        workflow2_env,
        override=False,
    )
# ============================================================
# POSTGRESQL
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

# ============================================================
# GCP / PUBSUB
# ============================================================

PROJECT_ID = os.getenv(
    "PROJECT_ID",
    "test-face-clustering",
)

PUBSUB_SUBSCRIPTION = os.getenv(
    "PUBSUB_SUBSCRIPTION",
    "wf2-face-extraction-sub",
)

PUBSUB_COMPLETION_TOPIC = os.getenv(
    "PUBSUB_COMPLETION_TOPIC",
    "wf1-batch-completed",
)