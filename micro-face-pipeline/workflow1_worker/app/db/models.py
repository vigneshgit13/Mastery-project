from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.postgres import Base


SCHEMA = "face_pipeline"


class Upload(Base):
    __tablename__ = "uploads"
    __table_args__ = (
        Index("idx_uploads_received_at", "received_at"),
        {"schema": SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    event_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False, unique=True)
    event_version: Mapped[str] = mapped_column(String(20), nullable=False)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    project_id: Mapped[str] = mapped_column(String(100), nullable=False)
    bucket: Mapped[str] = mapped_column(String(255), nullable=False)
    blob_name: Mapped[str] = mapped_column(Text, nullable=False)
    original_filename: Mapped[Optional[str]] = mapped_column(Text)
    content_type: Mapped[Optional[str]] = mapped_column(String(100))
    file_size: Mapped[Optional[int]] = mapped_column(BigInteger)
    uploaded_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_by: Mapped[Optional[str]] = mapped_column(String(100))
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    images = relationship("Image", back_populates="upload", cascade="all, delete-orphan")
    processing_jobs = relationship("ProcessingJob", back_populates="upload", cascade="all, delete-orphan")


class Image(Base):
    __tablename__ = "images"
    __table_args__ = (
        UniqueConstraint("bucket", "blob_name", name="uq_image_object"),
        Index("idx_images_upload_id", "upload_id"),
        {"schema": SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    upload_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(f"{SCHEMA}.uploads.id", ondelete="CASCADE"),
        nullable=False,
    )
    bucket: Mapped[str] = mapped_column(String(255), nullable=False)
    blob_name: Mapped[str] = mapped_column(Text, nullable=False)
    local_path: Mapped[Optional[str]] = mapped_column(Text)
    content_type: Mapped[Optional[str]] = mapped_column(String(100))
    file_size: Mapped[Optional[int]] = mapped_column(BigInteger)
    width: Mapped[Optional[int]] = mapped_column(Integer)
    height: Mapped[Optional[int]] = mapped_column(Integer)
    channels: Mapped[Optional[int]] = mapped_column(SmallInteger)
    image_hash: Mapped[Optional[str]] = mapped_column(String(128))
    downloaded_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    upload = relationship("Upload", back_populates="images")
    faces = relationship("Face", back_populates="image", cascade="all, delete-orphan")


class Face(Base):
    __tablename__ = "faces"
    __table_args__ = (
        UniqueConstraint("image_id", "face_index", name="uq_face_per_image"),
        Index("idx_faces_image_id", "image_id"),
        Index("idx_faces_gender_age", "gender", "age_group"),
        {"schema": SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    image_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(f"{SCHEMA}.images.id", ondelete="CASCADE"),
        nullable=False,
    )
    face_index: Mapped[int] = mapped_column(Integer, nullable=False)
    detection_confidence: Mapped[float] = mapped_column(Float, nullable=False)

    bbox_x1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_x2: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y2: Mapped[float] = mapped_column(Float, nullable=False)

    landmark_1_x: Mapped[Optional[float]] = mapped_column(Float)
    landmark_1_y: Mapped[Optional[float]] = mapped_column(Float)
    landmark_2_x: Mapped[Optional[float]] = mapped_column(Float)
    landmark_2_y: Mapped[Optional[float]] = mapped_column(Float)
    landmark_3_x: Mapped[Optional[float]] = mapped_column(Float)
    landmark_3_y: Mapped[Optional[float]] = mapped_column(Float)
    landmark_4_x: Mapped[Optional[float]] = mapped_column(Float)
    landmark_4_y: Mapped[Optional[float]] = mapped_column(Float)
    landmark_5_x: Mapped[Optional[float]] = mapped_column(Float)
    landmark_5_y: Mapped[Optional[float]] = mapped_column(Float)

    gender: Mapped[Optional[str]] = mapped_column(String(30))
    gender_confidence: Mapped[Optional[float]] = mapped_column(Float)
    age_years: Mapped[Optional[int]] = mapped_column(Integer)
    age_group: Mapped[Optional[str]] = mapped_column(String(30))
    age_confidence: Mapped[Optional[float]] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    image = relationship("Image", back_populates="faces")
    embedding = relationship("FaceEmbedding", back_populates="face", uselist=False, cascade="all, delete-orphan")


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"
    __table_args__ = (
        UniqueConstraint("face_id", name="uq_face_embedding"),
        Index("idx_face_embeddings_vector_id", "vector_id"),
        {"schema": SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    face_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(f"{SCHEMA}.faces.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[Optional[str]] = mapped_column(String(100))
    embedding_dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    vector_store: Mapped[str] = mapped_column(String(50), nullable=False)
    vector_id: Mapped[str] = mapped_column(Text, nullable=False)
    normalized: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default="true")
    embedding_norm: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    face = relationship("Face", back_populates="embedding")


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"
    __table_args__ = (
        UniqueConstraint("upload_id", "workflow", name="uq_processing_job"),
        Index("idx_processing_jobs_status", "status"),
        Index("idx_processing_jobs_workflow_status", "workflow", "status"),
        {"schema": SCHEMA},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    upload_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(f"{SCHEMA}.uploads.id", ondelete="CASCADE"),
        nullable=False,
    )
    workflow: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    error_code: Mapped[Optional[str]] = mapped_column(String(100))
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    upload = relationship("Upload", back_populates="processing_jobs")


class Workflow1OutboxEvent(Base):
    __tablename__ = "workflow1_outbox_events"
    __table_args__ = (
        UniqueConstraint(
            "event_id",
            name="uq_workflow1_outbox_event_id",
        ),
        Index(
            "idx_workflow1_outbox_status",
            "status",
        ),
        Index(
            "idx_workflow1_outbox_created_at",
            "created_at",
        ),
        {"schema": SCHEMA},
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    event_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    event_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    aggregate_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    aggregate_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    payload: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="PENDING",
        server_default="PENDING",
    )

    attempt_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    last_error: Mapped[Optional[str]] = mapped_column(
        Text,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    published_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )