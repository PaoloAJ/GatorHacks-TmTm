"""
Configuration settings for the application
"""
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import List, Optional


class Settings(BaseSettings):
    """Application settings"""

    model_config = ConfigDict(
        env_file=".env",
        protected_namespaces=('settings_',)
    )

    # API Settings
    app_name: str = "Style Similarity API"
    version: str = "1.0.0"
    api_prefix: str = "/api/v1"

    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000

    # CORS Settings
    # Allow localhost for development and all Vercel deployments
    cors_origins: List[str] = ["http://localhost:3000"]
    cors_origin_regex: Optional[str] = r"https://.*\.vercel\.app"

    # Model Settings
    embedding_dim: int = 512
    use_pretrained: bool = True
    model_device: str = "auto"  # "auto", "cpu", or "cuda"

    # Pinecone Settings
    pinecone_api_key: str = ""
    pinecone_index_name: str = "artist-styles"


settings = Settings()
