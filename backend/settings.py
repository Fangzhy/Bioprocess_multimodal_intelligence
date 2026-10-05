"""Environment-backed service settings."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    api_token: str | None = os.getenv("BIOPROCESS_API_TOKEN")
    max_upload_bytes: int = 5_000_000
    max_images: int = 6


settings = Settings()
