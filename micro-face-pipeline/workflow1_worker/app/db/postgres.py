import os
import logging
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


logger = logging.getLogger(__name__)


DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL environment variable is not set."
    )


class Base(DeclarativeBase):
    pass


engine = create_engine(
    DATABASE_URL,

    # Production-safe connection pooling.
    pool_size=int(
        os.getenv("DB_POOL_SIZE", "10")
    ),

    max_overflow=int(
        os.getenv("DB_MAX_OVERFLOW", "20")
    ),

    pool_timeout=int(
        os.getenv("DB_POOL_TIMEOUT", "30")
    ),

    pool_recycle=int(
        os.getenv("DB_POOL_RECYCLE", "1800")
    ),

    pool_pre_ping=True,

    future=True,
)


SessionLocal = sessionmaker(
    bind=engine,
    class_=Session,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


def check_database_connection() -> None:
    """
    Verify that PostgreSQL is reachable.
    """

    from sqlalchemy import text

    with engine.connect() as connection:
        connection.execute(
            text("SELECT 1")
        )

    logger.info(
        "PostgreSQL connection successful"
    )