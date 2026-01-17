"""
Application configuration using Pydantic Settings.
Load from environment variables or .env file.
"""

from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # Application
    app_name: str = "Entity Resolution API"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: str = "development"
    
    # API
    api_prefix: str = "/api/v1"
    allowed_origins: str = "http://localhost:3000,http://localhost:5173"
    
    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/entity_resolution"
    database_url_sync: str = "postgresql://postgres:postgres@localhost:5432/entity_resolution"
    db_pool_size: int = 5
    db_max_overflow: int = 10
    
    # Redis (for Celery)
    redis_url: str = "redis://localhost:6379/0"
    
    # Authentication (Phase 3)
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24  # 24 hours
    
    # Google OAuth
    google_client_id: Optional[str] = None
    google_client_secret: Optional[str] = None
    
    # GitHub OAuth
    github_client_id: Optional[str] = None
    github_client_secret: Optional[str] = None
    
    # OAuth Redirect
    oauth_redirect_uri: str = "http://localhost:8000/api/v1/auth/callback"
    frontend_url: str = "http://localhost:3000"
    
    # Matching Configuration
    match_threshold: float = 0.7  # Minimum score to consider a match
    max_results: int = 100  # Maximum results per query
    blocking_enabled: bool = True
    
    # Bulk Processing
    max_bulk_records: int = 500
    bulk_chunk_size: int = 50
    
    @property
    def allowed_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",")]


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
