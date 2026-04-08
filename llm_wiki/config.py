"""Configuration management via pydantic-settings."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="LLM_WIKI_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM backend
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI-compatible API base URL",
    )
    llm_api_key: str = Field(
        default="",
        description="API key for the LLM backend",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="Model name to use for completions",
    )
    llm_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    llm_max_tokens: int = Field(
        default=4096,
        gt=0,
        description="Maximum tokens per completion",
    )

    # Vault
    vault_path: Path = Field(
        default=Path("vault"),
        description="Path to the wiki vault directory",
    )

    # Behavior
    dry_run: bool = Field(
        default=False,
        description="Preview actions without writing files",
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose output",
    )

    @field_validator("vault_path", mode="before")
    @classmethod
    def resolve_vault_path(cls, v: Union[str, Path]) -> Path:
        return Path(v)

    def is_llm_configured(self) -> bool:
        return bool(self.llm_api_key)

    def display_dict(self) -> dict:
        """Return settings safe to display (API key masked)."""
        d = self.model_dump()
        if d.get("llm_api_key"):
            key = d["llm_api_key"]
            d["llm_api_key"] = key[:8] + "..." if len(key) > 8 else "***"
        d["vault_path"] = str(d["vault_path"])
        return d


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset cached settings (useful for testing)."""
    global _settings
    _settings = None
