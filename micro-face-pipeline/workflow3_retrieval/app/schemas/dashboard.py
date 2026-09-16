from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class DistributionItem(BaseModel):
    label: str
    count: int = Field(ge=0)


class DashboardSummaryResponse(BaseModel):
    total_faces: int = Field(ge=0)
    total_clusters: int = Field(ge=0)
    total_images: int = Field(ge=0)

    gender_distribution: list[DistributionItem]
    age_group_distribution: list[DistributionItem]


class ClusterSummary(BaseModel):
    cluster_id: int
    created_at: datetime
    updated_at: datetime
    face_count: int = Field(ge=0)


class ClusterListResponse(BaseModel):
    clusters: list[ClusterSummary]
    total_clusters: int = Field(ge=0)


class ClusterFaceDetail(BaseModel):
    face_id: int
    image_id: int
    face_index: int

    detection_confidence: float | None = None

    gender: str | None = None
    gender_confidence: float | None = None

    age_years: int | None = None
    age_group: str | None = None
    age_confidence: float | None = None

    similarity: float

    bucket: str
    blob_name: str


class ClusterDetailResponse(BaseModel):
    cluster_id: int
    created_at: datetime
    updated_at: datetime
    face_count: int = Field(ge=0)

    faces: list[ClusterFaceDetail]


class ClusterImage(BaseModel):
    image_id: int
    upload_id: int

    bucket: str
    blob_name: str

    content_type: str | None = None
    file_size: int | None = None

    width: int | None = None
    height: int | None = None
    channels: int | None = None

    image_hash: str | None = None
    created_at: datetime


class ClusterImagesResponse(BaseModel):
    cluster_id: int
    images: list[ClusterImage]
    total_images: int = Field(ge=0)


class ImageFace(BaseModel):
    face_id: int
    image_id: int
    face_index: int

    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float

    detection_confidence: float | None = None

    gender: str | None = None
    gender_confidence: float | None = None

    age_years: int | None = None
    age_group: str | None = None
    age_confidence: float | None = None


class ImageFacesResponse(BaseModel):
    image_id: int
    faces: list[ImageFace]
    total_faces: int = Field(ge=0)