"""
SQLAlchemy models for the entity resolution database.
"""

from datetime import date, datetime
from typing import Optional
from sqlalchemy import (
    Column, String, Date, DateTime, Index, Text,
    func, text
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
import uuid


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class Person(Base):
    """
    Person entity for matching.
    Contains PII fields used for entity resolution.
    """
    __tablename__ = "persons"
    
    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    
    # Core identity fields
    first_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    middle_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    last_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    
    # Date fields
    birth_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True, index=True)
    death_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    
    # SSN (masked/synthetic format: XXX-XX-1234 - only last 4 visible)
    ssn_last4: Mapped[Optional[str]] = mapped_column(String(4), nullable=True)
    ssn_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    
    # Location
    city: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    state: Mapped[Optional[str]] = mapped_column(String(2), nullable=True, index=True)
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )
    source_system: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    
    # Indexes for efficient blocking and matching
    __table_args__ = (
        # Composite index for common blocking strategies
        Index('idx_person_blocking_name_dob', 'last_name', 'birth_date'),
        Index('idx_person_blocking_state_name', 'state', 'last_name'),
        
        # Note: Trigram indexes must be created via raw SQL after pg_trgm extension
        # See alembic migration for GIN indexes
    )
    
    def __repr__(self) -> str:
        return f"<Person {self.first_name} {self.last_name} ({self.id})>"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for Splink processing."""
        return {
            "id": str(self.id),
            "first_name": self.first_name,
            "middle_name": self.middle_name,
            "last_name": self.last_name,
            "birth_date": self.birth_date.isoformat() if self.birth_date else None,
            "death_date": self.death_date.isoformat() if self.death_date else None,
            "ssn_last4": self.ssn_last4,
            "city": self.city,
            "state": self.state,
        }


class MatchJob(Base):
    """
    Track bulk matching jobs for async processing.
    """
    __tablename__ = "match_jobs"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    
    # Job status
    status: Mapped[str] = mapped_column(
        String(20),
        default="pending"  # pending, processing, completed, failed
    )
    
    # Progress tracking
    total_records: Mapped[int] = mapped_column(default=0)
    processed_records: Mapped[int] = mapped_column(default=0)
    matched_records: Mapped[int] = mapped_column(default=0)
    
    # File info
    original_filename: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    file_format: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    
    # Results storage (JSON)
    results_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # User tracking (for Phase 3)
    user_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    def __repr__(self) -> str:
        return f"<MatchJob {self.id} ({self.status})>"
