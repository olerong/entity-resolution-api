"""
Entity Resolution API - Main Application
Fast, accurate entity matching using hybrid deterministic and probabilistic methods.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.database import init_database, create_indexes
from app.api import match, bulk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - startup and shutdown."""
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")
    
    # Initialize database
    try:
        await init_database()
        await create_indexes()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        # Continue anyway - database might already exist
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="""
## Entity Resolution API

High-performance entity matching service supporting:

- **Single Record Matching**: Match individual records in real-time
- **Bulk Upload Processing**: Upload CSV, JSON, Excel, Parquet, XML, or TXT files (up to 500 records)
- **Multiple Matching Algorithms**:
  - Deterministic Exact: Strict field matching
  - Deterministic Fuzzy: Levenshtein/similarity-based matching
  - Probabilistic: Splink-based Fellegi-Sunter model
  - Hybrid: Best of both worlds (recommended)

### Supported Fields
- First Name, Middle Name, Last Name
- Birth Date
- SSN (last 4 digits)
- City, State

### Authentication
OAuth 2.0 via Google or GitHub (coming soon)
    """,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
app.include_router(
    match.router,
    prefix=f"{settings.api_prefix}/match",
    tags=["Matching"],
)

app.include_router(
    bulk.router,
    prefix=f"{settings.api_prefix}/bulk",
    tags=["Bulk Upload"],
)

app.include_router(
    bulk.router,
    prefix=f"{settings.api_prefix}/seed",
    tags=["Seed"],
)

@app.get(f"{settings.api_prefix}/seed-demo", tags=["Seed"])
async def seed_demo():
    """Seed database with demo data."""
    from faker import Faker
    import random
    from app.core.database import AsyncSessionLocal
    from app.models.database import Person
    
    fake = Faker()
    
    async with AsyncSessionLocal() as session:
        from sqlalchemy import text
        result = await session.execute(text("SELECT COUNT(*) FROM persons"))
        count = result.scalar()
        if count > 100:
            return {"message": f"Database already has {count} records"}
        
        for _ in range(1000):
            person = Person(
                first_name=fake.first_name(),
                last_name=fake.last_name(),
                birth_date=fake.date_of_birth(minimum_age=18, maximum_age=90),
                city=fake.city(),
                state=fake.state_abbr(),
                ssn_last4=str(random.randint(1000, 9999)),
            )
            session.add(person)
        
        await session.commit()
    
    return {"message": "Seeded 1000 demo records"}
    
# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": f"{settings.api_prefix}/match/health",
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )
