from typing import List, Optional

from pydantic import BaseModel


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
    """

    face_index: int

    # Detection
    detection_confidence: float
    bbox: List[float]

    # Recognition
    embedding: List[float]

    # Attributes
    gender: GenderResult
    age: AgeResult

    # Source
    image_path: Optional[str] = None

    # Future clustering fields
    cluster_id: Optional[str] = None
    person_id: Optional[str] = None