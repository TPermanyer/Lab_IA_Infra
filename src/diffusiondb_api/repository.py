"""Acceso a los metadatos de DiffusionDB con DuckDB sobre el parquet.

Todas las consultas usan parámetros enlazados (nunca concatenan entrada del usuario).
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from diffusiondb_api.schemas import SAMPLER_NAMES, CountItem, DatasetStats, PromptRecord

COLUMNS = (
    "image_name, prompt, part_id, seed, step, cfg, sampler, width, height, "
    "user_name, timestamp, image_nsfw, prompt_nsfw"
)
TOP_RESOLUTIONS = 10


@dataclass(frozen=True)
class PromptFilters:
    q: str | None = None
    sampler: int | None = None
    min_step: int | None = None
    max_step: int | None = None
    min_cfg: float | None = None
    max_cfg: float | None = None
    width: int | None = None
    height: int | None = None
    max_image_nsfw: float | None = None
    max_prompt_nsfw: float | None = None


class DiffusionDBRepository:
    def __init__(self, parquet_path: Path):
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"No se encuentra {parquet_path}. Ejecuta: uv run python -m diffusiondb_api.download"
            )
        # Los timestamps del dataset son TIMESTAMPTZ: se devuelven siempre en UTC.
        # Va en `config` (no en SET) para que lo hereden los cursores.
        self._conn = duckdb.connect(database=":memory:", config={"TimeZone": "UTC"})
        # Vista sobre el parquet: DuckDB lo lee bajo demanda, sin cargarlo entero en memoria.
        path_literal = str(parquet_path.resolve()).replace("'", "''")
        self._conn.execute(f"CREATE VIEW metadata AS SELECT * FROM read_parquet('{path_literal}')")

    def close(self) -> None:
        self._conn.close()

    def _query(self, sql: str, params: list[Any] | None = None) -> list[tuple]:
        # Un cursor por consulta: las conexiones DuckDB no son seguras entre hilos.
        cur = self._conn.cursor()
        try:
            return cur.execute(sql, params or []).fetchall()
        finally:
            cur.close()

    def search(self, filters: PromptFilters, limit: int, offset: int) -> tuple[list[PromptRecord], int]:
        where, params = _build_where(filters)
        total = self._query(f"SELECT count(*) FROM metadata {where}", params)[0][0]
        rows = self._query(
            f"SELECT {COLUMNS} FROM metadata {where} ORDER BY image_name LIMIT ? OFFSET ?",
            [*params, limit, offset],
        )
        return [_to_record(r) for r in rows], total

    def get(self, image_name: str) -> PromptRecord | None:
        rows = self._query(f"SELECT {COLUMNS} FROM metadata WHERE image_name = ?", [image_name])
        return _to_record(rows[0]) if rows else None

    def stats(self) -> DatasetStats:
        total, prompts, users, avg_step, median_cfg, first, last = self._query(
            "SELECT count(*), count(DISTINCT prompt), count(DISTINCT user_name), "
            # cfg tiene NaN/inf y valores absurdos (negativos, hasta 2e28): mediana, no media.
            "coalesce(avg(step), 0), coalesce(median(cfg) FILTER (WHERE isfinite(cfg)), 0), "
            "min(timestamp), max(timestamp) FROM metadata"
        )[0]
        by_sampler = self._query(
            "SELECT sampler, count(*) AS n FROM metadata GROUP BY sampler ORDER BY n DESC"
        )
        resolutions = self._query(
            "SELECT width || 'x' || height AS res, count(*) AS n FROM metadata "
            "GROUP BY res ORDER BY n DESC LIMIT ?",
            [TOP_RESOLUTIONS],
        )
        return DatasetStats(
            total_images=total,
            unique_prompts=prompts,
            unique_users=users,
            avg_step=round(avg_step, 2),
            median_cfg=round(median_cfg, 2),
            first_timestamp=first,
            last_timestamp=last,
            by_sampler=[CountItem(key=_sampler_name(s), count=n) for s, n in by_sampler],
            top_resolutions=[CountItem(key=r, count=n) for r, n in resolutions],
        )


def _build_where(f: PromptFilters) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []

    def add(clause: str, value: Any) -> None:
        clauses.append(clause)
        params.append(value)

    if f.q:
        add("prompt ILIKE ? ESCAPE '\\'", f"%{_escape_like(f.q)}%")
    if f.sampler is not None:
        add("sampler = ?", f.sampler)
    if f.min_step is not None:
        add("step >= ?", f.min_step)
    if f.max_step is not None:
        add("step <= ?", f.max_step)
    # isfinite: en DuckDB NaN es mayor que cualquier número y colaría en min_cfg.
    if f.min_cfg is not None:
        add("isfinite(cfg) AND cfg >= ?", f.min_cfg)
    if f.max_cfg is not None:
        add("isfinite(cfg) AND cfg <= ?", f.max_cfg)
    if f.width is not None:
        add("width = ?", f.width)
    if f.height is not None:
        add("height = ?", f.height)
    if f.max_image_nsfw is not None:
        add("image_nsfw <= ?", f.max_image_nsfw)
    if f.max_prompt_nsfw is not None:
        add("prompt_nsfw <= ?", f.max_prompt_nsfw)

    return ("WHERE " + " AND ".join(clauses) if clauses else ""), params


def _escape_like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _sampler_name(code: int) -> str:
    return SAMPLER_NAMES.get(code, "unknown")


def _to_record(row: tuple) -> PromptRecord:
    (image_name, prompt, part_id, seed, step, cfg, sampler, width, height,
     user_name, timestamp, image_nsfw, prompt_nsfw) = row
    return PromptRecord(
        image_name=image_name,
        prompt=prompt,
        part_id=part_id,
        seed=seed,
        step=step,
        cfg=cfg if cfg is not None and math.isfinite(cfg) else None,
        sampler=sampler,
        sampler_name=_sampler_name(sampler),
        width=width,
        height=height,
        user_name=user_name,
        timestamp=timestamp,
        image_nsfw=image_nsfw,
        prompt_nsfw=prompt_nsfw,
    )
