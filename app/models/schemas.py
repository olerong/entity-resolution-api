"""
Pydantic schemas for API request/response validation.
"""

from datetime import date, datetime
from typing import Optional, List, Any
from pydantic import BaseModel, Field, field_validator
from uuid import UUID
from enum import Enum


# ============== Enums ==============

class MatchType(str, Enum):
    """Types of matching algorithms available."""
    DETERMINISTIC_EXACT = "deterministic_exact"
    DETERMINISTIC_FUZZY = "deterministic_fuzzy"
    PROBABILISTIC = "probabilistic"
    HYBRID = "hybrid"  # Combines deterministic blocking with probabilistic scoring


class JobStatus(str, Enum):
    """Status of a bulk matching job."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class FileFormat(str, Enum):
    """Supported file formats for bulk upload."""
    CSV = "csv"
    JSON = "json"
    XLSX = "xlsx"
    TXT = "txt"
    PARQUET = "parquet"
    XML = "xml"


# ============== Input Schemas ==============

class RecordInput(BaseModel):
    """Input schema for a single record to match."""
    first_name: str = Field(..., min_length=1, max_length=100, description="First name")
    middle_name: Optional[str] = Field(None, max_length=100, description="Middle name")
    last_name: str = Field(..., min_length=1, max_length=100, description="Last name")
    birth_date: Optional[date] = Field(None, description="Date of birth (YYYY-MM-DD)")
    ssn_last4: Optional[str] = Field(None, min_length=4, max_length=4, pattern=r"^\d{4}$", description="Last 4 digits of SSN")
    city: Optional[str] = Field(None, max_length=100, description="City")
    state: Optional[str] = Field(None, min_length=2, max_length=2, description="State code (2 letters)")
    
    @field_validator("state")
    @classmethod
    def validate_state(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            return v.upper()
        return v
    
    @field_validator("first_name", "last_name", "city")
    @classmethod
    def normalize_name(cls, v: str) -> str:
        if v:
            return v.strip().title()
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "first_name": "John",
                "middle_name": "Michael",
                "last_name": "Smith",
                "birth_date": "1985-03-15",
                "ssn_last4": "1234",
                "city": "New York",
                "state": "NY"
            }
        }


class MatchRequest(BaseModel):
    """Request schema for matching operation."""
    record: RecordInput
    match_type: MatchType = Field(default=MatchType.HYBRID, description="Matching algorithm to use")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum match score (0-1)")
    max_results: Optional[int] = Field(None, ge=1, le=500, description="Maximum results to return")
    include_scores: bool = Field(default=True, description="Include detailed match scores")


class BulkMatchRequest(BaseModel):
    """Request schema for bulk matching operation."""
    match_type: MatchType = Field(default=MatchType.HYBRID)
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_results_per_record: Optional[int] = Field(default=10, ge=1, le=100)


# ============== Output Schemas ==============

class PersonRecord(BaseModel):
    """Schema for a person record in results."""
    id: UUID
    first_name: str
    middle_name: Optional[str] = None
    last_name: str
    birth_date: Optional[date] = None
    death_date: Optional[date] = None
    ssn_last4: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    
    class Config:
        from_attributes = True


class FieldScore(BaseModel):
    """Detailed score breakdown for a single field."""
    field: str
    score: float
    match_level: str  # exact, fuzzy, partial, no_match
    weight: float


class MatchResult(BaseModel):
    """Single match result with scores."""
    record: PersonRecord
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall match confidence")
    match_probability: Optional[float] = Field(None, description="Probabilistic match probability")
    field_scores: Optional[List[FieldScore]] = Field(None, description="Per-field score breakdown")
    match_type_used: MatchType


class MatchResponse(BaseModel):
    """Response schema for single record matching."""
    query: RecordInput
    matches: List[MatchResult]
    total_matches: int
    processing_time_ms: float
    match_type: MatchType
    threshold_used: float


class BulkRecordResult(BaseModel):
    """Result for a single record in bulk processing."""
    row_number: int
    input_record: dict
    matches: List[MatchResult]
    match_count: int
    best_score: Optional[float] = None


class JobResponse(BaseModel):
    """Response schema for job status."""
    job_id: UUID
    status: JobStatus
    total_records: int
    processed_records: int
    matched_records: int
    progress_percent: float
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    class Config:
        from_attributes = True


class BulkMatchResponse(BaseModel):
    """Response for completed bulk matching."""
    job_id: UUID
    status: JobStatus
    total_records: int
    records_with_matches: int
    total_matches_found: int
    processing_time_seconds: float
    results: Optional[List[BulkRecordResult]] = None
    download_urls: Optional[dict] = None  # {"csv": "...", "json": "..."}


# ============== Health & Status ==============

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    database: str
    timestamp: datetime


class StatsResponse(BaseModel):
    """Database statistics response."""
    total_records: int
    records_with_death_date: int
    unique_states: int
    index_status: dict
