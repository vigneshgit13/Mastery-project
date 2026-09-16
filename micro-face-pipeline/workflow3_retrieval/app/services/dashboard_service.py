from __future__ import annotations

from app.repositories.dashboard_repository import DashboardRepository


class DashboardService:
    """
    Application service for Workflow 3 dashboard operations.

    The service coordinates repository calls and prepares data
    for the API layer.
    """

    def __init__(
        self,
        repository: DashboardRepository,
    ) -> None:
        self.repository = repository

    def get_summary(self) -> dict:
        """
        Return the dashboard summary.
        """

        return self.repository.get_summary()

    def get_clusters(self) -> list[dict]:
        return self.repository.get_clusters()

    def get_cluster_detail(self, cluster_id: int) -> dict | None:
        return self.repository.get_cluster_detail(cluster_id)

    def get_cluster_images(self, cluster_id: int) -> list[dict]:
        return self.repository.get_cluster_images(cluster_id)

    def get_image_faces(self, image_id: int) -> dict | None:
        faces = self.repository.get_image_faces(image_id)

        if not faces:
            return None

        return {
           "image_id": image_id,
           "faces": faces,
           "total_faces": len(faces),
                }