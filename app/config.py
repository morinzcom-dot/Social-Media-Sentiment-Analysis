"""
الإعدادات المركزية للمشروع
يُقرأ من ملف .env تلقائياً
"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # ── قاعدة البيانات ──
    DATABASE_URL: str = "sqlite+aiosqlite:///./sentiment_ai.db"
    # للإنتاج: DATABASE_URL = "postgresql+asyncpg://user:pass@localhost/sentimentai"

    # ── النموذج ──
    # "arabert"  → يستخدم aubmindlab/bert-base-arabertv2
    # "logistic" → يستخدم Logistic Regression (خفيف وسريع)
    MODEL_TYPE: str = "logistic"
    ARABERT_MODEL_NAME: str = "aubmindlab/bert-base-arabertv2"

    # ── Facebook API ──
    FACEBOOK_APP_ID: str = ""
    FACEBOOK_APP_SECRET: str = ""
    FACEBOOK_ACCESS_TOKEN: str = ""

    # ── الأمان ──
    SECRET_KEY: str = "change-this-secret-key-in-production"
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    # ── عام ──
    MAX_TEXT_LENGTH: int = 512
    BATCH_SIZE: int = 32

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
