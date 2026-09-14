from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class GenderResult(BaseModel):
    label: str
    confidence: float


class AgeResult(BaseModel):
    years: int
    group: str
    confidence: Optional[float] = None


class FaceRecord(BaseModel):
    """
    Complete AI result for one detected face.

    This is the contract between the AI pipeline and the
    Workflow 1 persistence/orchestration layer.
    """

    face_index: int

    # ============================================================
    # Detection
    # ============================================================

    detection_confidence: float

    # [x1, y1, x2, y2]
    bbox: List[float] = Field(
        min_length=4,
        max_length=4,
    )

    # Five SCRFD landmarks flattened as:
    #
    # [
    #   left_eye_x,  left_eye_y,
    #   right_eye_x, right_eye_y,
    #   nose_x,      nose_y,
    #   left_mouth_x, left_mouth_y,
    #   right_mouth_x, right_mouth_y,
    # ]
    #
    # Stored as 10 values because PostgreSQL has ten landmark columns.
    landmarks: Optional[List[float]] = Field(
        default=None,
        min_length=10,
        max_length=10,
    )

    # ============================================================
    # Recognition
    # ============================================================

    # ArcFace 512-D embedding
    embedding: List[float]

    # ============================================================
    # Attributes
    # ============================================================

    gender: GenderResult
    age: AgeResult

    # ============================================================
    # Source
    # ============================================================

    image_path: Optional[str] = None

    # ============================================================
    # Future clustering fields
    # ============================================================

    cluster_id: Optional[str] = None
    person_id: Optional[str] = None