from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.postgres import get_db
from app.repositories.dashboard_repository import DashboardRepository
from app.schemas.dashboard import (
    DashboardSummaryResponse,
    DistributionItem,
    ClusterListResponse,
    ClusterDetailResponse,
    ClusterImagesResponse,
    ImageFacesResponse,
)
from app.services.dashboard_service import DashboardService




router = APIRouter(
    prefix="/api/dashboard",
    tags=["dashboard"],
)


@router.get(
    "/summary",
    response_model=DashboardSummaryResponse,
)
def get_dashboard_summary(
    db: Session = Depends(get_db),
) -> DashboardSummaryResponse:
    repository = DashboardRepository(db)
    service = DashboardService(repository)

    summary = service.get_summary()

    return DashboardSummaryResponse(
        total_faces=summary["total_faces"],
        total_clusters=summary["total_clusters"],
        total_images=summary["total_images"],
        gender_distribution=[
            DistributionItem(label=label, count=count)
            for label, count in summary["gender_distribution"].items()
        ],
        age_group_distribution=[
            DistributionItem(label=label, count=count)
            for label, count in summary["age_group_distribution"].items()
        ],
    )

@router.get(
    "/clusters",
    response_model=ClusterListResponse,
)
def get_clusters(
    db: Session = Depends(get_db),
) -> ClusterListResponse:
    repository = DashboardRepository(db)
    service = DashboardService(repository)

    clusters = service.get_clusters()

    return ClusterListResponse(
        clusters=clusters,
        total_clusters=len(clusters),
    )


@router.get(
    "/clusters/{cluster_id}",
    response_model=ClusterDetailResponse,
)
def get_cluster_detail(
    cluster_id: int,
    db: Session = Depends(get_db),
) -> ClusterDetailResponse:
    repository = DashboardRepository(db)
    service = DashboardService(repository)

    cluster = service.get_cluster_detail(cluster_id)

    if cluster is None:
        raise HTTPException(
            status_code=404,
            detail=f"Cluster {cluster_id} not found",
        )

    return ClusterDetailResponse(
        **cluster
    )

@router.get(
    "/clusters/{cluster_id}/images",
    response_model=ClusterImagesResponse,
)
def get_cluster_images(
    cluster_id: int,
    db: Session = Depends(get_db),
) -> ClusterImagesResponse:
    repository = DashboardRepository(db)
    service = DashboardService(repository)

    # First verify that the cluster exists.
    cluster = service.get_cluster_detail(cluster_id)

    if cluster is None:
        raise HTTPException(
            status_code=404,
            detail=f"Cluster {cluster_id} not found",
        )

    images = service.get_cluster_images(cluster_id)

    return ClusterImagesResponse(
        cluster_id=cluster_id,
        images=images,
        total_images=len(images),
    )

@router.get(
    "/images/{image_id}/faces",
    response_model=ImageFacesResponse,
)
def get_image_faces(
    image_id: int,
    db: Session = Depends(get_db),
) -> ImageFacesResponse:
    repository = DashboardRepository(db)
    service = DashboardService(repository)

    result = service.get_image_faces(image_id)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Image {image_id} not found",
        )

    return ImageFacesResponse(**result)