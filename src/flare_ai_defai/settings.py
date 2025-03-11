# """
# Settings Configuration Module

# This module defines the configuration settings for the AI Agent API
# using Pydantic's BaseSettings. It handles environment variables and
# provides default values for various service configurations.

# The settings can be overridden by environment variables or through a .env file.
# Environment variables take precedence over values defined in the .env file.
# """

# import structlog
# from pydantic_settings import BaseSettings, SettingsConfigDict

# logger = structlog.get_logger(__name__)


# class Settings(BaseSettings):
#     """
#     Application settings model that provides configuration for all components.
#     """

#     # Flag to enable/disable attestation simulation
#     simulate_attestation: bool = False
#     # Restrict backend listener to specific IPs
#     cors_origins: list[str] = ["*"]
#     # API key for accessing Google's Gemini AI service
#     gemini_api_key: str = ""
#     # The Gemini model identifier to use
#     gemini_model: str = "gemini-1.5-flash"
#     # API version to use at the backend
#     api_version: str = "v1"
#     # URL for the Flare Network RPC provider
#     web3_provider_url: str = "https://coston2-api.flare.network/ext/C/rpc"
#     # URL for the Flare Network block explorer
#     web3_explorer_url: str = "https://coston2-explorer.flare.network/"

#     model_config = SettingsConfigDict(
#         # This enables .env file support
#         env_file=".env",
#         # If .env file is not found, don't raise an error
#         env_file_encoding="utf-8",
#         # Optional: you can also specify multiple .env files
#         extra="ignore",
#     )


# # Create a global settings instance
# settings = Settings()
# logger.debug("settings", settings=settings.model_dump())



# UNCOMMENT FROM HERE

from pathlib import Path
from typing import Literal, TypedDict

import structlog
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from flare_ai_consensus.settings import ConsensusConfig, ModelConfig, AggregatorConfig

logger = structlog.get_logger(__name__)


def create_path(folder_name: str) -> Path:
    """Creates and returns a path for storing data or logs."""
    path = Path(__file__).parent.resolve().parent / f"{folder_name}"
    path.mkdir(exist_ok=True)
    return path


class Message(TypedDict):
    role: str
    content: str


class Settings(BaseSettings):
    """
    Application settings model that provides configuration for all components.
    Combines both infrastructure and consensus settings.
    """

    simulate_attestation: bool = False
    # Restrict backend listener to specific IPs
    cors_origins: list[str] = ["*"]
    # API key for accessing Google's Gemini AI service
    gemini_api_key: str = ""
    # The Gemini model identifier to use
    gemini_model: str = "gemini-1.5-flash"
    # API version to use at the backend
    api_version: str = "v1"
    # URL for the Flare Network RPC provider
    web3_provider_url: str = "https://coston2-api.flare.network/ext/C/rpc"
    # URL for the Flare Network block explorer
    web3_explorer_url: str = "https://coston2-explorer.flare.network/"


    # OpenRouter Settings
    open_router_base_url: str = "https://openrouter.ai/api/v1"
    open_router_api_key: str = ""
    gemini_embedding_key: str = ""

    # Path Settings
    data_path: Path = create_path("data")
    input_path: Path = create_path("flare_ai_consensus")

    # Restrict backend listener to specific IPs
    cors_origins: list[str] = ["*"]

    # Consensus Settings
    consensus_config: ConsensusConfig | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    print(model_config)

    def load_consensus_config(self, json_data: dict) -> None:
        """Load consensus configuration from JSON data"""
        self.consensus_config = ConsensusConfig.from_json(json_data)
        logger.info("loaded consensus configuration")


# Create a global settings instance
settings = Settings()
logger.debug("settings initialized", settings=settings.model_dump())

