from app.services.redis_service import RedisService


def main() -> None:
    print("=" * 60)
    print("REDIS SERVICE VERIFICATION")
    print("=" * 60)

    redis_service = RedisService()

    # ------------------------------------------------------------
    # 1. Connection
    # ------------------------------------------------------------

    print("\n[1] Testing Redis connection...")

    if not redis_service.ping():
        raise RuntimeError(
            "Redis connection failed."
        )

    print("Redis connection: PASS")

    # ------------------------------------------------------------
    # Test key
    # ------------------------------------------------------------

    key = "workflow1:test"

    # ------------------------------------------------------------
    # 2. SET
    # ------------------------------------------------------------

    print("\n[2] Testing SET...")

    result = redis_service.set(
        key,
        {
            "workflow": "workflow1",
            "image_id": 123,
            "status": "processing",
        },
    )

    print("SET result:", result)

    if not result:
        raise AssertionError(
            "SET failed."
        )

    print("SET check: PASS")

    # ------------------------------------------------------------
    # 3. EXISTS
    # ------------------------------------------------------------

    print("\n[3] Testing EXISTS...")

    exists = redis_service.exists(key)

    print("Exists:", exists)

    if not exists:
        raise AssertionError(
            "Key should exist."
        )

    print("EXISTS check: PASS")

    # ------------------------------------------------------------
    # 4. GET
    # ------------------------------------------------------------

    print("\n[4] Testing GET...")

    value = redis_service.get(key)

    print("Value:", value)

    expected = {
        "workflow": "workflow1",
        "image_id": 123,
        "status": "processing",
    }

    if value != expected:
        raise AssertionError(
            f"Unexpected value: {value}"
        )

    print("GET check: PASS")

    # ------------------------------------------------------------
    # 5. TTL
    # ------------------------------------------------------------

    print("\n[5] Testing TTL...")

    redis_service.set(
        "workflow1:test:ttl",
        {
            "temporary": True,
        },
        ttl=60,
    )

    ttl = redis_service.ttl(
        "workflow1:test:ttl"
    )

    print("TTL:", ttl)

    if ttl <= 0:
        raise AssertionError(
            "TTL was not applied."
        )

    print("TTL check: PASS")

    # ------------------------------------------------------------
    # 6. HASH SET
    # ------------------------------------------------------------

    print("\n[6] Testing HSET...")

    redis_service.hset(
        "workflow1:faces",
        "face:101",
        {
            "face_id": 101,
            "image_id": 1,
            "gender": "male",
            "age_group": "adult",
        },
    )

    print("HSET check: PASS")

    # ------------------------------------------------------------
    # 7. HASH GET
    # ------------------------------------------------------------

    print("\n[7] Testing HGET...")

    face = redis_service.hget(
        "workflow1:faces",
        "face:101",
    )

    print("Face:", face)

    if face is None:
        raise AssertionError(
            "Face metadata was not found."
        )

    if face["face_id"] != 101:
        raise AssertionError(
            "Incorrect face_id."
        )

    print("HGET check: PASS")

    # ------------------------------------------------------------
    # 8. HASH GET ALL
    # ------------------------------------------------------------

    print("\n[8] Testing HGETALL...")

    faces = redis_service.hgetall(
        "workflow1:faces"
    )

    print("Faces:", faces)

    if "face:101" not in faces:
        raise AssertionError(
            "face:101 missing."
        )

    print("HGETALL check: PASS")

    # ------------------------------------------------------------
    # 9. HASH DELETE
    # ------------------------------------------------------------

    print("\n[9] Testing HDELETE...")

    deleted = redis_service.hdelete(
        "workflow1:faces",
        "face:101",
    )

    print("Deleted:", deleted)

    if not deleted:
        raise AssertionError(
            "Hash field was not deleted."
        )

    print("HDELETE check: PASS")

    # ------------------------------------------------------------
    # 10. Cleanup
    # ------------------------------------------------------------

    print("\n[10] Cleaning up...")

    redis_service.delete(key)
    redis_service.delete(
        "workflow1:test:ttl"
    )
    redis_service.delete(
        "workflow1:faces"
    )

    if redis_service.exists(key):
        raise AssertionError(
            "Cleanup failed."
        )

    print("Cleanup check: PASS")

    redis_service.close()

    # ------------------------------------------------------------
    # Final
    # ------------------------------------------------------------

    print()
    print("=" * 60)
    print("REDIS SERVICE VERIFICATION: PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()