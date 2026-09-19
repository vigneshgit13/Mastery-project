from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent


# ============================================================
# LOAD WORKFLOW 3 ENVIRONMENT
# ============================================================

workflow3_env = BASE_DIR / ".env"

if workflow3_env.exists():
    load_dotenv(
        workflow3_env,
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



CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://127.0.0.1:3000,http://localhost:3000",
    ).split(",")
    if origin.strip()
]