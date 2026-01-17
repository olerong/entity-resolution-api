"""
Database connection and session management.
Uses asyncpg for async operations and psycopg2 for Splink compatibility.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
import logging

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from app.core.config import get_settings
from app.models.database import Base

logger = logging.getLogger(__name__)
settings = get_settings()


# ============== Async Engine (for FastAPI routes) ==============

async_engine = create_async_engine(
    settings.database_url,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    echo=settings.debug,
    pool_pre_ping=True,  # Verify connections before use
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for async database sessions."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ============== Sync Engine (for Splink/DuckDB operations) ==============

sync_engine = create_engine(
    settings.database_url_sync,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    echo=settings.debug,
    pool_pre_ping=True,
)

SyncSessionLocal = sessionmaker(
    bind=sync_engine,
    autocommit=False,
    autoflush=False,
)


def get_sync_session() -> Session:
    """Get a synchronous database session for Splink operations."""
    return SyncSessionLocal()


# ============== Database Initialization ==============

async def init_database() -> None:
    """Initialize database tables and extensions."""
    async with async_engine.begin() as conn:
        # Create extensions
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm;"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;"))
        
        # Create tables
        await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Database initialized with extensions and tables")


async def create_indexes() -> None:
    """Create optimized indexes for entity resolution."""
    async with async_engine.begin() as conn:
        # Trigram indexes for fuzzy name matching
        index_statements = [
            """
            CREATE INDEX IF NOT EXISTS idx_persons_first_name_trgm 
            ON persons USING GIN (first_name gin_trgm_ops);
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_persons_last_name_trgm 
            ON persons USING GIN (last_name gin_trgm_ops);
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_persons_city_trgm 
            ON persons USING GIN (city gin_trgm_ops);
            """,
            # Soundex index for phonetic blocking
            """
            CREATE INDEX IF NOT EXISTS idx_persons_soundex_last 
            ON persons (soundex(last_name));
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_persons_soundex_first 
            ON persons (soundex(first_name));
            """,
            # Blocking key index (first 2 chars of last name + birth year)
            """
            CREATE INDEX IF NOT EXISTS idx_persons_blocking_key 
            ON persons (
                UPPER(LEFT(last_name, 2)),
                EXTRACT(YEAR FROM birth_date)
            );
            """,
        ]
        
        for stmt in index_statements:
            try:
                await conn.execute(text(stmt))
            except Exception as e:
                logger.warning(f"Index creation warning: {e}")
        
        logger.info("Optimized indexes created")


async def check_database_health() -> dict:
    """Check database connectivity and return status."""
    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(text("SELECT 1"))
            result.scalar()
            
            # Check extensions
            ext_result = await session.execute(
                text("SELECT extname FROM pg_extension WHERE extname IN ('pg_trgm', 'fuzzystrmatch')")
            )
            extensions = [row[0] for row in ext_result.fetchall()]
            
            # Get record count
            count_result = await session.execute(text("SELECT COUNT(*) FROM persons"))
            record_count = count_result.scalar()
            
            return {
                "status": "healthy",
                "extensions": extensions,
                "record_count": record_count,
            }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
        }


@asynccontextmanager
async def get_db_context():
    """Context manager for database sessions."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
