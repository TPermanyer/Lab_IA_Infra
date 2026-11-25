"""Configuración leída de variables de entorno (prefijo DIFFUSIONDB_)."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

DATASET_BASE_URL = "https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main"
METADATA_FILES = {
    "2m": "metadata.parquet",
    "large": "metadata-large.parquet",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DIFFUSIONDB_", env_file=".env")

    data_dir: Path = Path("data")
    variant: Literal["2m", "large"] = "2m"

    @property
    def parquet_path(self) -> Path:
        return self.data_dir / METADATA_FILES[self.variant]


def get_settings() -> Settings:
    return Settings()
