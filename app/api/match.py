"""
API routes for entity resolution matching.
"""

import logging
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_async_session, check_database_health
from app.models.schemas import (
    RecordInput, MatchRequest, MatchResponse, MatchType,
    HealthResponse, StatsResponse
)
from app.services.matcher import EntityMatcher, get_matcher, MatchConfig

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API and database health."""
    db_status = await check_database_health()
    
    return HealthResponse(
        status="healthy" if db_status["status"] == "healthy" else "degraded",
        version=settings.app_version,
        database=db_status["status"],
        timestamp=datetime.utcnow(),
    )


@router.get("/stats", response_model=StatsResponse)
async def get_stats(
    session: AsyncSession = Depends(get_async_session),
):
    """Get database statistics."""
    from sqlalchemy import text
    
    # Get total records
    result = await session.execute(text("SELECT COUNT(*) FROM persons"))
    total_records = result.scalar()
    
    # Get records with death date
    result = await session.execute(
        text("SELECT COUNT(*) FROM persons WHERE death_date IS NOT NULL")
    )
    with_death = result.scalar()
    
    # Get unique states
    result = await session.execute(
        text("SELECT COUNT(DISTINCT state) FROM persons WHERE state IS NOT NULL")
    )
    unique_states = result.scalar()
    
    # Check index status
    result = await session.execute(text("""
        SELECT indexname, indexdef 
        FROM pg_indexes 
        WHERE tablename = 'persons'
    """))
    indexes = {row[0]: "active" for row in result.fetchall()}
    
    return StatsResponse(
        total_records=total_records or 0,
        records_with_death_date=with_death or 0,
        unique_states=unique_states or 0,
        index_status=indexes,
    )


@router.post("/match", response_model=MatchResponse)
async def match_record(
    request: MatchRequest,
    session: AsyncSession = Depends(get_async_session),
    matcher: EntityMatcher = Depends(get_matcher),
):
    """
    Match a single record against the database.
    
    Supports multiple matching algorithms:
    - deterministic_exact: Exact field matching only
    - deterministic_fuzzy: Fuzzy string matching (Levenshtein, etc.)
    - probabilistic: Splink-based probabilistic matching
    - hybrid: Combines fuzzy blocking with probabilistic scoring (recommended)
    """
    config = MatchConfig(
        match_type=request.match_type,
        threshold=request.threshold or settings.match_threshold,
        max_results=request.max_results or settings.max_results,
        include_scores=request.include_scores,
    )
    
    try:
        result = await matcher.match_single(
            session=session,
            record=request.record,
            config=config,
        )
        return result
    except Exception as e:
        logger.error(f"Matching error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Matching failed: {str(e)}",
        )


@router.post("/match/quick", response_model=MatchResponse)
async def quick_match(
    record: RecordInput,
    match_type: MatchType = MatchType.HYBRID,
    threshold: Optional[float] = None,
    max_results: Optional[int] = None,
    session: AsyncSession = Depends(get_async_session),
    matcher: EntityMatcher = Depends(get_matcher),
):
    """
    Quick match endpoint - simpler interface for single record matching.
    Takes record fields directly as parameters.
    """
    config = MatchConfig(
        match_type=match_type,
        threshold=threshold or settings.match_threshold,
        max_results=max_results or settings.max_results,
        include_scores=True,
    )
    
    try:
        result = await matcher.match_single(
            session=session,
            record=record,
            config=config,
        )
        return result
    except Exception as e:
        logger.error(f"Quick matching error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Matching failed: {str(e)}",
        )


@router.get("/match-types")
async def list_match_types():
    """List available matching algorithms with descriptions."""
    return {
        "match_types": [
            {
                "type": MatchType.DETERMINISTIC_EXACT,
                "name": "Deterministic Exact",
                "description": "Exact field matching only. Fast but strict.",
                "use_case": "When data quality is high and exact matches are needed.",
            },
            {
                "type": MatchType.DETERMINISTIC_FUZZY,
                "name": "Deterministic Fuzzy",
                "description": "Fuzzy string matching using similarity algorithms.",
                "use_case": "When there may be typos or slight variations in data.",
            },
            {
                "type": MatchType.PROBABILISTIC,
                "name": "Probabilistic (Splink)",
                "description": "Full probabilistic matching using Fellegi-Sunter model.",
                "use_case": "Best accuracy for complex matching scenarios.",
            },
            {
                "type": MatchType.HYBRID,
                "name": "Hybrid (Recommended)",
                "description": "Combines fuzzy blocking with probabilistic scoring.",
                "use_case": "Best balance of speed and accuracy for most use cases.",
            },
        ]
    }
