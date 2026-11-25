"""Modelos de respuesta de la API."""

from datetime import datetime
from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")

# Códigos de sampler según la ficha del dataset DiffusionDB.
SAMPLER_NAMES = {
    1: "ddim",
    2: "plms",
    3: "k_euler",
    4: "k_euler_ancestral",
    5: "k_heun",
    6: "k_dpm_2",
    7: "k_dpm_2_ancestral",
    8: "k_lms",
    9: "others",
}


class PromptRecord(BaseModel):
    image_name: str
    prompt: str | None
    part_id: int
    seed: int
    step: int
    cfg: float | None  # None si el dataset trae NaN/inf
    sampler: int
    sampler_name: str
    width: int
    height: int
    user_name: str | None
    timestamp: datetime | None
    image_nsfw: float | None
    prompt_nsfw: float | None


class PageMeta(BaseModel):
    total: int
    limit: int
    offset: int


class ApiResponse(BaseModel, Generic[T]):
    success: bool = True
    data: T | None = None
    error: str | None = None
    meta: PageMeta | None = None


class CountItem(BaseModel):
    key: str
    count: int


class DatasetStats(BaseModel):
    total_images: int
    unique_prompts: int
    unique_users: int
    avg_step: float
    median_cfg: float
    first_timestamp: datetime | None
    last_timestamp: datetime | None
    by_sampler: list[CountItem]
    top_resolutions: list[CountItem]
