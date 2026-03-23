"""
app/core/config.py
------------------
Centralized application configuration.
All settings are loaded from environment variables or a .env file.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Gemini API key from https://aistudio.google.com/apikey
    LLM_API_KEY: str = Field(..., env="LLM_API_KEY")

    # Sentence-Transformers model used for embeddings (runs locally, no API needed)
    EMBEDDING_MODEL: str = Field(
        default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL"
    )

    # Google Gemini chat model
    CHAT_MODEL: str = Field(
        default="llama-3.3-70b-versatile", env="CHAT_MODEL"
    )

    # Directory where FAISS index + metadata are persisted
    VECTOR_STORE_DIR: str = Field(
        default="data/vector_store", env="VECTOR_STORE_DIR"
    )

    # How many top chunks to retrieve per query
    TOP_K: int = Field(default=3, env="TOP_K")

    # Path to the raw knowledge-base text file
    RAW_DATA_PATH: str = Field(
        default="data/raw/campus_handbook.txt", env="RAW_DATA_PATH"
    )

    # Path where processed chunks JSON is saved
    PROCESSED_CHUNKS_PATH: str = Field(
        default="data/processed/chunks.json", env="PROCESSED_CHUNKS_PATH"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Single shared instance — import this everywhere
settings = Settings()
