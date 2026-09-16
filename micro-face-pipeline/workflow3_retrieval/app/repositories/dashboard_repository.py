from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.orm import Session


SCHEMA = "face_pipeline"


class DashboardRepository:
    """
    Read-only PostgreSQL repository for Workflow 3 dashboard data.

    PostgreSQL is the authoritative source for dashboard/entity state.
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    def count_faces(self) -> int:
        """Return the total number of detected faces."""

        query = text(
            f"""
            SELECT COUNT(*)
            FROM {SCHEMA}.faces
            """
        )

        return int(self.db.execute(query).scalar_one())

    def count_clusters(self) -> int:
        """Return the total number of face clusters."""

        query = text(
            f"""
            SELECT COUNT(*)
            FROM {SCHEMA}.face_clusters
            """
        )

        return int(self.db.execute(query).scalar_one())

    def count_images(self) -> int:
        """Return the total number of processed images."""

        query = text(
            f"""
            SELECT COUNT(*)
            FROM {SCHEMA}.images
            """
        )

        return int(self.db.execute(query).scalar_one())

    def get_gender_distribution(self) -> dict[str, int]:
        """
        Return face counts grouped by gender.

        NULL gender values are exposed as 'unknown'.
        """

        query = text(
            f"""
            SELECT
                COALESCE(NULLIF(TRIM(gender), ''), 'unknown') AS gender,
                COUNT(*) AS face_count
            FROM {SCHEMA}.faces
            GROUP BY COALESCE(NULLIF(TRIM(gender), ''), 'unknown')
            ORDER BY gender
            """
        )

        rows = self.db.execute(query).mappings().all()

        return {
            str(row["gender"]): int(row["face_count"])
            for row in rows
        }

    def get_age_group_distribution(self) -> dict[str, int]:
        """
        Return face counts grouped by age group.

        NULL or empty age_group values are exposed as 'unknown'.
        """

        query = text(
            f"""
            SELECT
                COALESCE(
                    NULLIF(TRIM(age_group), ''),
                    'unknown'
                ) AS age_group,
                COUNT(*) AS face_count
            FROM {SCHEMA}.faces
            GROUP BY COALESCE(
                NULLIF(TRIM(age_group), ''),
                'unknown'
            )
            ORDER BY age_group
            """
        )

        rows = self.db.execute(query).mappings().all()

        return {
            str(row["age_group"]): int(row["face_count"])
            for row in rows
        }

    def get_summary(self) -> dict:
        """
        Return the complete dashboard summary.
        """

        return {
            "total_faces": self.count_faces(),
            "total_clusters": self.count_clusters(),
            "total_images": self.count_images(),
            "gender_distribution": self.get_gender_distribution(),
            "age_group_distribution": self.get_age_group_distribution(),
        }

    def get_clusters(self) -> list[dict]:
        """
        Return all face clusters with their membership counts.
        """

        query = text(
            f"""
            SELECT
                fc.id AS cluster_id,
                fc.created_at,
                fc.updated_at,
                COUNT(fcm.face_id) AS face_count
            FROM {SCHEMA}.face_clusters fc
            LEFT JOIN {SCHEMA}.face_cluster_members fcm
                ON fcm.cluster_id = fc.id
            GROUP BY
                fc.id,
                fc.created_at,
                fc.updated_at
            ORDER BY fc.id
            """
        )

        rows = self.db.execute(query).mappings().all()

        return [
            {
                "cluster_id": row["cluster_id"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "face_count": int(row["face_count"]),
            }
            for row in rows
        ]

        def get_cluster_detail(self, cluster_id: int) -> dict | None:
            cluster_query = text(f"""
                        SELECT
                            fc.id AS cluster_id,
                            fc.created_at,
                            fc.updated_at
                        FROM {SCHEMA}.face_clusters fc
                        WHERE fc.id = :cluster_id
                    """)
            cluster = self.db.execute(
                                    cluster_query,
                                    {"cluster_id": cluster_id},
                                ).mappings().first()
            if cluster is None:
                return None


            faces_query = text(f"""
                        SELECT
                         fcm.face_id,
                         f.image_id,
                         f.face_index,
                         f.detection_confidence,
                         f.gender,
                         f.gender_confidence,
                         f.age_years,
                         f.age_group,
                         f.age_confidence,
                         fcm.similarity,
                         i.bucket,
                        i.blob_name
                        FROM {SCHEMA}.face_cluster_members fcm
                        JOIN {SCHEMA}.faces f
                        ON f.id = fcm.face_id
                        JOIN {SCHEMA}.images i
                        ON i.id = f.image_id
                        WHERE fcm.cluster_id = :cluster_id
                        ORDER BY fcm.face_id
                                            """)
                                                                                                            
                                                
                                                                                    
                                                                                       
            faces = self.db.execute(
                                      faces_query,
                                      {"cluster_id": cluster_id},
                                     ).mappings().all()                      
                                                
            return {
                     "cluster_id": cluster["cluster_id"],
                     "created_at": cluster["created_at"],
                     "updated_at": cluster["updated_at"],
                     "face_count": len(faces),
                     "faces": [
                         {
                             "face_id": row["face_id"],
                             "image_id": row["image_id"],
                             "face_index": row["face_index"],
                             "detection_confidence": row["detection_confidence"],
                             "gender": row["gender"],
                             "gender_confidence": row["gender_confidence"],
                             "age_years": row["age_years"],
                             "age_group": row["age_group"],
                             "age_confidence": row["age_confidence"],
                             "similarity": row["similarity"],
                             "bucket": row["bucket"],
                             "blob_name": row["blob_name"],
                         }
                        for row in faces
                                ],
                    }
    def get_cluster_detail(self, cluster_id: int) -> dict | None:
         cluster_query = text(f"""
            SELECT
                fc.id AS cluster_id,
                fc.created_at,
                fc.updated_at
            FROM {SCHEMA}.face_clusters fc
            WHERE fc.id = :cluster_id
        """)

         cluster = self.db.execute(
            cluster_query,
            {"cluster_id": cluster_id},
        ).mappings().first()

         if cluster is None:
             return None  


         faces_query = text(f"""
            SELECT
                fcm.face_id,
                f.image_id,
                f.face_index,
                f.detection_confidence,
                f.gender,
                f.gender_confidence,
                f.age_years,
                f.age_group,
                f.age_confidence,
                fcm.similarity,
                i.bucket,
                i.blob_name
            FROM {SCHEMA}.face_cluster_members fcm
            JOIN {SCHEMA}.faces f
                ON f.id = fcm.face_id
            JOIN {SCHEMA}.images i
                ON i.id = f.image_id
            WHERE fcm.cluster_id = :cluster_id
            ORDER BY fcm.face_id
        """)

         faces = self.db.execute(
            faces_query,
            {"cluster_id": cluster_id},
        ).mappings().all()

         return {
            "cluster_id": cluster["cluster_id"],
            "created_at": cluster["created_at"],
            "updated_at": cluster["updated_at"],
            "face_count": len(faces),
            "faces": [
                {
                    "face_id": row["face_id"],
                    "image_id": row["image_id"],
                    "face_index": row["face_index"],
                    "detection_confidence": row["detection_confidence"],
                    "gender": row["gender"],
                    "gender_confidence": row["gender_confidence"],
                    "age_years": row["age_years"],
                    "age_group": row["age_group"],
                    "age_confidence": row["age_confidence"],
                    "similarity": row["similarity"],
                    "bucket": row["bucket"],
                    "blob_name": row["blob_name"],
                }
                for row in faces
            ],
        }                              
    def get_cluster_images(self, cluster_id: int) -> list[dict]:
        query = text(f"""
            SELECT DISTINCT
                i.id AS image_id,
                i.upload_id,
                i.bucket,
                i.blob_name,
                i.content_type,
                i.file_size,
                i.width,
                i.height,
                i.channels,
                i.image_hash,
                i.created_at
            FROM {SCHEMA}.face_cluster_members fcm
            JOIN {SCHEMA}.faces f
                ON f.id = fcm.face_id
            JOIN {SCHEMA}.images i
                ON i.id = f.image_id
            WHERE fcm.cluster_id = :cluster_id
            ORDER BY i.id
        """)

        rows = self.db.execute(
            query,
            {"cluster_id": cluster_id},
        ).mappings().all()

        return [
            {
                "image_id": row["image_id"],
                "upload_id": row["upload_id"],
                "bucket": row["bucket"],
                "blob_name": row["blob_name"],
                "content_type": row["content_type"],
                "file_size": row["file_size"],
                "width": row["width"],
                "height": row["height"],
                "channels": row["channels"],
                "image_hash": row["image_hash"],
                "created_at": row["created_at"],
            }
            for row in rows
        ] 
                                        
    def get_image_faces(self, image_id: int) -> list[dict]:
        query = text(f"""
             SELECT
                f.id AS face_id,
                f.image_id,
                f.face_index,
                f.bbox_x1,
                f.bbox_y1,
                f.bbox_x2,
                f.bbox_y2,
                f.detection_confidence,
                f.gender,
                f.gender_confidence,
                f.age_years,
                f.age_group,
                f.age_confidence
            FROM {SCHEMA}.faces f
            WHERE f.image_id = :image_id
            ORDER BY f.face_index
        """)
        rows = self.db.execute(
            query,
            {"image_id": image_id},
        ).mappings().all()

        return [
            {
                "face_id": row["face_id"],
                "image_id": row["image_id"],
                "face_index": row["face_index"],
                "bbox_x1": row["bbox_x1"],
                "bbox_y1": row["bbox_y1"],
                "bbox_x2": row["bbox_x2"],
                "bbox_y2": row["bbox_y2"],
                "detection_confidence": row["detection_confidence"],
                "gender": row["gender"],
                "gender_confidence": row["gender_confidence"],
                "age_years": row["age_years"],
                "age_group": row["age_group"],
                "age_confidence": row["age_confidence"],
            }
            for row in rows
        ]

                            
                    
                    