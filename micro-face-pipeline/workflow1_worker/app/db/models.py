import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Index,
    func,
)

from sqlalchemy.dialects.postgresql import UUID, JSONB

from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class Upload(Base):
    __tablename__ = "uploads"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    event_id: Mapped[str | None] = mapped_column(
        String(255),
        unique=True,
        nullable=True,
        index=True,
    )

    bucket_name: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
    )

    blob_name: Mapped[str] = mapped_column(
        String(1024),
        nullable=False,
    )

    original_filename: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="RECEIVED",
        index=True,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    images = relationship(
        "Image",
        back_populates="upload",
        cascade="all, delete-orphan",
    )


class Image(Base):
    __tablename__ = "images"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    upload_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("uploads.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    gcs_uri: Mapped[str] = mapped_column(
        String(2048),
        nullable=False,
    )

    width: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    height: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="PROCESSING",
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    processed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    upload = relationship(
        "Upload",
        back_populates="images",
    )

    faces = relationship(
        "Face",
        back_populates="image",
        cascade="all, delete-orphan",
    )


class Face(Base):
    __tablename__ = "faces"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    image_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("images.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    face_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    bbox: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
    )

    detection_confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    gender: Mapped[str | None] = mapped_column(
        String(30),
        nullable=True,
    )

    gender_confidence: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    age_years: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    age_group: Mapped[str | None] = mapped_column(
        String(30),
        nullable=True,
    )

    age_confidence: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    embedding_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    embedding_dimension: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    image = relationship(
        "Image",
        back_populates="faces",
    )

    __table_args__ = (
        UniqueConstraint(
            "image_id",
            "face_index",
            name="uq_face_image_index",
        ),

        Index(
            "ix_faces_gender_age",
            "gender",
            "age_group",
        ),
    )


class Person(Base):
    """
    Unique person / cluster.

    This table will be populated by Workflow 2.
    Workflow 1 only creates Face records.
    """

    __tablename__ = "persons"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    gender: Mapped[str | None] = mapped_column(
        String(30),
        nullable=True,
    )

    age_group: Mapped[str | None] = mapped_column(
        String(30),
        nullable=True,
    )

    face_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    image_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="ACTIVE",
    )

    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )