from __future__ import annotations

import logging
from typing import Generator

from sqlalchemy import URL, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import (
    POSTGRES_DATABASE,
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_PORT,
    POSTGRES_USER,
)


logger = logging.getLogger(__name__)


# ============================================================
# Validation
# ============================================================

if not POSTGRES_PASSWORD:
    raise RuntimeError(
        "POSTGRES_PASSWORD environment variable is not set."
    )


# ============================================================
# SQLAlchemy database URL
# ============================================================

DATABASE_URL = URL.create(
    drivername="postgresql+psycopg",
    username=POSTGRES_USER,
    password=POSTGRES_PASSWORD,
    host=POSTGRES_HOST,
    port=POSTGRES_PORT,
    database=POSTGRES_DATABASE,
)


# ============================================================
# SQLAlchemy Base
# ============================================================

class Base(DeclarativeBase):
    pass


# ============================================================
# SQLAlchemy Engine
# ============================================================

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=1800,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
)


# ============================================================
# Session Factory
# ============================================================

SessionLocal = sessionmaker(
    bind=engine,
    class_=Session,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


# ============================================================
# Database Session
# ============================================================

def get_db() -> Generator[Session, None, None]:
    """
    Create and yield a SQLAlchemy database session.

    The session is always closed after use.
    """

    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


# ============================================================
# Connection Verification
# ============================================================

def check_database_connection() -> None:
    """
    Verify that SQLAlchemy can connect to PostgreSQL.
    """

    with engine.connect() as connection:
        connection.execute(
            text("SELECT 1")
        )

    logger.info(
        "PostgreSQL connection successful"
    )