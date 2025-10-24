"""
Configuration settings for the application
"""
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import List


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
    cors_origins: List[str] = ["http://localhost:3000"]

    # Model Settings
    embedding_dim: int = 512
    use_pretrained: bool = True
    model_device: str = "auto"  # "auto", "cpu", or "cuda"


settings = Settings()
