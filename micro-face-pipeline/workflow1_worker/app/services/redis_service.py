"""
Redis service for Workflow 1.

Redis is used for fast, temporary operational metadata.

PostgreSQL remains the authoritative persistent database.
FAISS/HNSW remains the vector-search store.

Typical Redis data:

    upload:{upload_id}
    image:{image_id}
    face:{face_id}

The service is intentionally small for the micro-version.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import redis


logger = logging.getLogger(__name__)


class RedisService:
    """
    Thread-safe Redis service wrapper.

    Values are stored as JSON so that dictionaries/lists and
    primitive values can be handled consistently.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        decode_responses: bool = True,
    ) -> None:
        self.host = host
        self.port = port
        self.db = db
        self.password = password

        self.client = redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            decode_responses=decode_responses,
        )

    # ================================================================
    # CONNECTION
    # ================================================================

    def ping(self) -> bool:
        """
        Verify Redis connectivity.
        """

        try:
            result = self.client.ping()

            if result:
                logger.info("Redis connection: OK")

            return bool(result)

        except redis.RedisError:
            logger.exception(
                "Redis connection failed."
            )
            return False

    # ================================================================
    # SET
    # ================================================================

    def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> bool:
        """
        Store a value.

        Parameters
        ----------
        key:
            Redis key.

        value:
            Any JSON-serializable value.

        ttl:
            Optional expiration time in seconds.
        """

        if not key:
            raise ValueError(
                "Redis key must not be empty."
            )

        serialized = json.dumps(
            value,
            separators=(",", ":"),
        )

        try:
            if ttl is not None:

                if ttl <= 0:
                    raise ValueError(
                        "ttl must be greater than zero."
                    )

                return bool(
                    self.client.set(
                        key,
                        serialized,
                        ex=ttl,
                    )
                )

            return bool(
                self.client.set(
                    key,
                    serialized,
                )
            )

        except redis.RedisError:
            logger.exception(
                "Failed to set Redis key: %s",
                key,
            )
            raise

    # ================================================================
    # GET
    # ================================================================

    def get(
        self,
        key: str,
    ) -> Any | None:
        """
        Retrieve and deserialize a value.

        Returns None when the key doesn't exist.
        """

        if not key:
            raise ValueError(
                "Redis key must not be empty."
            )

        try:
            value = self.client.get(key)

            if value is None:
                return None

            return json.loads(value)

        except redis.RedisError:
            logger.exception(
                "Failed to get Redis key: %s",
                key,
            )
            raise

        except json.JSONDecodeError:
            logger.exception(
                "Invalid JSON stored at Redis key: %s",
                key,
            )
            raise

    # ================================================================
    # DELETE
    # ================================================================

    def delete(
        self,
        key: str,
    ) -> bool:
        """
        Delete a Redis key.

        Returns True when a key was deleted.
        """

        if not key:
            raise ValueError(
                "Redis key must not be empty."
            )

        try:
            return (
                self.client.delete(key) > 0
            )

        except redis.RedisError:
            logger.exception(
                "Failed to delete Redis key: %s",
                key,
            )
            raise

    # ================================================================
    # EXISTS
    # ================================================================

    def exists(
        self,
        key: str,
    ) -> bool:
        """
        Check whether a Redis key exists.
        """

        if not key:
            raise ValueError(
                "Redis key must not be empty."
            )

        try:
            return bool(
                self.client.exists(key)
            )

        except redis.RedisError:
            logger.exception(
                "Failed to check Redis key: %s",
                key,
            )
            raise

    # ================================================================
    # EXPIRE
    # ================================================================

    def expire(
        self,
        key: str,
        ttl: int,
    ) -> bool:
        """
        Set/update expiration time for an existing key.
        """

        if not key:
            raise ValueError(
                "Redis key must not be empty."
            )

        if ttl <= 0:
            raise ValueError(
                "ttl must be greater than zero."
            )

        try:
            return bool(
                self.client.expire(
                    key,
                    ttl,
                )
            )

        except redis.RedisError:
            logger.exception(
                "Failed to set expiration for key: %s",
                key,
            )
            raise

    # ================================================================
    # TTL
    # ================================================================

    def ttl(
        self,
        key: str,
    ) -> int:
        """
        Return remaining TTL in seconds.

        Redis semantics:
            -1 = key exists without expiration
            -2 = key does not exist
        """

        if not key:
            raise ValueError(
                "Redis key must not be empty."
            )

        try:
            return int(
                self.client.ttl(key)
            )

        except redis.RedisError:
            logger.exception(
                "Failed to retrieve TTL for key: %s",
                key,
            )
            raise

    # ================================================================
    # HASH
    # ================================================================

    def hset(
        self,
        key: str,
        field: str,
        value: Any,
    ) -> bool:
        """
        Store one field in a Redis hash.
        """

        if not key:
            raise ValueError(
                "Redis key must not be empty."
            )

        if not field:
            raise ValueError(
                "Redis hash field must not be empty."
            )

        serialized = json.dumps(
            value,
            separators=(",", ":"),
        )

        try:
            self.client.hset(
                key,
                field,
                serialized,
            )

            return True

        except redis.RedisError:
            logger.exception(
                "Failed to HSET %s[%s]",
                key,
                field,
            )
            raise

    # ================================================================
    # HASH GET
    # ================================================================

    def hget(
        self,
        key: str,
        field: str,
    ) -> Any | None:
        """
        Retrieve one field from a Redis hash.
        """

        if not key:
            raise ValueError(
                "Redis key must not be empty."
            )

        if not field:
            raise ValueError(
                "Redis hash field must not be empty."
            )

        try:
            value = self.client.hget(
                key,
                field,
            )

            if value is None:
                return None

            return json.loads(value)

        except redis.RedisError:
            logger.exception(
                "Failed to HGET %s[%s]",
                key,
                field,
            )
            raise

        except json.JSONDecodeError:
            logger.exception(
                "Invalid JSON in %s[%s]",
                key,
                field,
            )
            raise

    # ================================================================
    # HASH GET ALL
    # ================================================================

    def hgetall(
        self,
        key: str,
    ) -> dict[str, Any]:
        """
        Retrieve and deserialize an entire Redis hash.
        """

        if not key:
            raise ValueError(
                "Redis key must not be empty."
            )

        try:
            values = self.client.hgetall(key)

            return {
                field: json.loads(value)
                for field, value
                in values.items()
            }

        except redis.RedisError:
            logger.exception(
                "Failed to HGETALL %s",
                key,
            )
            raise

        except json.JSONDecodeError:
            logger.exception(
                "Invalid JSON found in Redis hash: %s",
                key,
            )
            raise

    # ================================================================
    # HASH DELETE
    # ================================================================

    def hdelete(
        self,
        key: str,
        field: str,
    ) -> bool:
        """
        Delete one field from a Redis hash.
        """

        if not key:
            raise ValueError(
                "Redis key must not be empty."
            )

        if not field:
            raise ValueError(
                "Redis hash field must not be empty."
            )

        try:
            return (
                self.client.hdel(
                    key,
                    field,
                )
                > 0
            )

        except redis.RedisError:
            logger.exception(
                "Failed to HDEL %s[%s]",
                key,
                field,
            )
            raise

    # ================================================================
    # CLEAR
    # ================================================================

    def clear(
        self,
        key: str,
    ) -> bool:
        """
        Delete a complete Redis key/hash.
        """

        return self.delete(key)

    # ================================================================
    # CLOSE
    # ================================================================

    def close(self) -> None:
        """
        Close the Redis client connection pool.
        """

        try:
            self.client.close()

        except Exception:
            logger.exception(
                "Error while closing Redis client."
            )