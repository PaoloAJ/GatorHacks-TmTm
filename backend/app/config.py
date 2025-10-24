from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    ENV: str = "dev"  # dev, staging, prod
    DATABASE_URL: str | None = None

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    settings = Settings()

    if settings.ENV == "prod":
        if not settings.DATABASE_URL:
            raise ValueError("DATABASE_URL must be set in production!")
    else:
        # Default: use SQLite in local environment
        settings.DATABASE_URL = "sqlite:///./local.db"

    return settings

settings = get_settings()
