import logging

from app.db.postgres import Base, engine
from app.db import models  # noqa: F401


logging.basicConfig(
    level=logging.INFO
)


def main():

    logging.info(
        "Creating PostgreSQL tables..."
    )

    Base.metadata.create_all(
        bind=engine
    )

    logging.info(
        "Database initialization complete."
    )


if __name__ == "__main__":
    main()