from pathlib import Path

from google.cloud import storage

from app.core.config import DOWNLOAD_DIR


class GCSService:
    """
    Downloads uploaded images from Google Cloud Storage.
    """

    def __init__(self):

        self.client = storage.Client()

    def download_file(
        self,
        bucket_name: str,
        blob_name: str,
    ) -> Path:
        """
        Download a blob from GCS to the local download directory.

        Args:
            bucket_name: GCS bucket name.
            blob_name: Object path inside the bucket.

        Returns:
            Local path of the downloaded file.
        """

        # Ensure download directory exists
        DOWNLOAD_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        bucket = self.client.bucket(bucket_name)

        blob = bucket.blob(blob_name)

        destination = DOWNLOAD_DIR / Path(blob_name).name

        blob.download_to_filename(str(destination))

        return destination


gcs_service = GCSService()