from __future__ import annotations

import requests


BASE_URL = "http://127.0.0.1:8001"


def check_status(path: str, expected: int) -> None:
    response = requests.get(
        BASE_URL + path,
        timeout=10,
    )

    actual = response.status_code

    if actual != expected:
        raise AssertionError(
            f"{path}: expected {expected}, got {actual}\n"
            f"Response: {response.text}"
        )

    print(f"PASS  {path}  [{actual}]")


def main() -> None:
    print("=" * 70)
    print("WORKFLOW 3 API REGRESSION TEST")
    print("=" * 70)

    print("\n[1] Health")
    check_status("/health", 200)

    print("\n[2] Dashboard APIs")
    check_status("/api/dashboard/summary", 200)
    check_status("/api/dashboard/clusters", 200)
    check_status("/api/dashboard/clusters/1", 200)
    check_status("/api/dashboard/clusters/1/images", 200)
    check_status("/api/dashboard/images/1/faces", 200)

    print("\n[3] Thumbnail API")
    check_status("/api/dashboard/faces/105/thumbnail", 200)

    print("\n[4] Thumbnail regression: faces 86-105")
    for face_id in range(86, 106):
        check_status(
            f"/api/dashboard/faces/{face_id}/thumbnail",
            200,
        )

    print("\n[5] 404 protection")
    check_status("/api/dashboard/clusters/999999", 404)
    check_status("/api/dashboard/clusters/999999/images", 404)
    check_status("/api/dashboard/images/999999/faces", 404)
    check_status("/api/dashboard/faces/999999/thumbnail", 404)

    print("\n" + "=" * 70)
    print("WORKFLOW 3 API REGRESSION: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()