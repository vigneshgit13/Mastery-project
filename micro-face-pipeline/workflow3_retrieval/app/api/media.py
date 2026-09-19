from __future__ import annotations

from io import BytesIO
from pathlib import Path

import cv2
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from sqlalchemy import text

from app.db.postgres import SessionLocal

router = APIRouter(prefix="/api/dashboard", tags=["dashboard-media"])
SCHEMA = "face_pipeline"


def _resolve_image_path(local_path: str | None) -> Path | None:
    """Resolve the persisted Workflow 1 image path without trusting a client path."""
    if not local_path:
        return None

    candidates = [Path(local_path)]
    source_name = Path(local_path).name

    # The project was moved from OneDrive to D:, so support the current
    # Workflow 1 temp directory when the historical absolute path is stale.
    project_root = Path(__file__).resolve().parents[3]
    candidates.append(project_root / "workflow1_worker" / "temp" / source_name)

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


@router.get("/faces/{face_id}/thumbnail")
def get_face_thumbnail(face_id: int) -> Response:
    """Return a cropped JPEG thumbnail for one detected face."""
    db = SessionLocal()
    try:
        query = text(f"""
            SELECT
                f.id AS face_id,
                f.bbox_x1,
                f.bbox_y1,
                f.bbox_x2,
                f.bbox_y2,
                i.local_path
            FROM {SCHEMA}.faces f
            JOIN {SCHEMA}.images i ON i.id = f.image_id
            WHERE f.id = :face_id
            LIMIT 1
        """)
        row = db.execute(query, {"face_id": int(face_id)}).mappings().first()
    finally:
        db.close()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Face {face_id} not found")

    image_path = _resolve_image_path(row["local_path"])
    if image_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Source image for face {face_id} is not available locally",
        )

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=422, detail="Unable to read source image")

    height, width = image.shape[:2]
    x1 = max(0, min(width - 1, int(round(float(row["bbox_x1"])))) )
    y1 = max(0, min(height - 1, int(round(float(row["bbox_y1"])))) )
    x2 = max(x1 + 1, min(width, int(round(float(row["bbox_x2"])))) )
    y2 = max(y1 + 1, min(height, int(round(float(row["bbox_y2"])))) )

    # Add context around the detected face so the thumbnail is not overly tight.
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(round(bw * 0.25))
    pad_y = int(round(bh * 0.25))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise HTTPException(status_code=422, detail="Face crop is empty")

    max_side = 320
    h, w = crop.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        crop = cv2.resize(
            crop,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    ok, encoded = cv2.imencode(
        ".jpg",
        crop,
        [int(cv2.IMWRITE_JPEG_QUALITY), 88],
    )
    if not ok:
        raise HTTPException(status_code=500, detail="Unable to encode face thumbnail")

    return Response(content=BytesIO(encoded.tobytes()).getvalue(), media_type="image/jpeg")
