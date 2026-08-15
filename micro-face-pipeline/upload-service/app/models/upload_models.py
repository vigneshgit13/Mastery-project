from pydantic import BaseModel


class UploadMetadata(BaseModel):
    original_filename: str
    content_type: str
    file_size: int


class UploadResponse(BaseModel):
    upload_id: str
    original_filename: str
    content_type: str
    file_size: int

    bucket: str
    blob_name: str

    uploaded_at: str

    status: str