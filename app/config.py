"""Configuration module for SATYA AI using Pydantic Settings."""

from functools import lru_cache
from typing import List, Optional, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    APP_NAME: str = Field(default="SATYA AI", description="Application name")
    APP_ENV: str = Field(
        default="development",
        description="Runtime environment: development, staging, production, test",
    )
    DEBUG: bool = Field(default=True, description="Debug mode flag")
    HOST: str = Field(default="127.0.0.1", description="Server host binding")
    PORT: int = Field(default=8000, description="Server port")
    API_V1_STR: str = Field(default="/api", description="API route prefix")

    # Database configuration (defaults to SQLite for Phase 1)
    DATABASE_URL: str = Field(
        default="sqlite:///./satya.db",
        description="SQLAlchemy database connection URI",
    )

    # Logging
    LOG_LEVEL: str = Field(default="INFO", description="Structured log level")

    # LLM Provider Configuration (Optional)
    LLM_PROVIDER: str = Field(
        default="none",
        description="Configured LLM provider: 'openai', 'gemini', 'anthropic', 'generic', or 'none'",
    )
    LLM_API_KEY: Optional[str] = Field(
        default=None,
        description="API key for the configured LLM provider",
    )
    LLM_BASE_URL: Optional[str] = Field(
        default=None,
        description="Base URL for custom/compatible LLM endpoints",
    )
    LLM_MODEL: str = Field(
        default="gpt-4o-mini",
        description="Model identifier to use for extraction/analysis",
    )

    # Retrieval Providers Configuration
    GOOGLE_FACT_CHECK_API_KEY: Optional[str] = Field(
        default=None,
        description="Google Fact Check Tools API key",
    )
    SEARCH_PROVIDER: str = Field(
        default="mock",
        description="Web search provider: 'tavily', 'serper', 'brave', 'duckduckgo', 'mock', or 'none'",
    )
    SEARCH_API_KEY: Optional[str] = Field(
        default=None,
        description="API key for the external search provider",
    )
    TAVILY_API_KEY: Optional[str] = Field(
        default=None,
        description="Direct alias for Tavily search API key",
    )
    SERPER_API_KEY: Optional[str] = Field(
        default=None,
        description="Direct alias for Serper search API key",
    )
    RETRIEVAL_TIMEOUT_SECONDS: float = Field(
        default=6.0,
        description="Timeout for external API calls and source fetches in seconds",
    )

    # WhatsApp Cloud API Configuration
    WHATSAPP_ACCESS_TOKEN: Optional[str] = Field(
        default=None,
        description="Meta WhatsApp Cloud API System User or User Access Token",
    )
    WHATSAPP_PHONE_NUMBER_ID: Optional[str] = Field(
        default=None,
        description="WhatsApp Cloud API Phone Number ID",
    )
    WHATSAPP_BUSINESS_ACCOUNT_ID: Optional[str] = Field(
        default=None,
        description="WhatsApp Business Account ID from Meta Dashboard",
    )
    WHATSAPP_VERIFY_TOKEN: Optional[str] = Field(
        default=None,
        description="Custom secret verification token configured in Meta App Dashboard for webhook setup",
    )
    WHATSAPP_APP_SECRET: Optional[str] = Field(
        default=None,
        description="Meta App Secret for HMAC-SHA256 signature verification of incoming webhooks",
    )
    META_APP_SECRET: Optional[str] = Field(
        default=None,
        description="Alias for Meta App Secret",
    )
    WHATSAPP_GRAPH_API_VERSION: str = Field(
        default="v21.0",
        description="Meta Graph API Version (e.g. 'v21.0')",
    )

    @property
    def effective_meta_app_secret(self) -> Optional[str]:
        """Resolve app secret from META_APP_SECRET or WHATSAPP_APP_SECRET."""
        return self.META_APP_SECRET or self.WHATSAPP_APP_SECRET

    # CORS configuration
    CORS_ORIGINS: Union[List[str], str] = Field(
        default=["*"],
        description="Allowed CORS origins list or string",
    )

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            if v.startswith("[") and v.endswith("]"):
                import json

                try:
                    return json.loads(v)
                except Exception:
                    pass
            return [i.strip() for i in v.split(",") if i.strip()]
        return v


@lru_cache()
def get_settings() -> Settings:
    """Return cached instance of application settings."""
    return Settings()


def reload_settings() -> Settings:
    """Clear cached settings and reload fresh from environment and .env file."""
    get_settings.cache_clear()
    return get_settings()

