from pathlib import Path

import duckdb
import pytest
from fastapi.testclient import TestClient

from diffusiondb_api.config import Settings
from diffusiondb_api.main import create_app

# Filas de ejemplo con el mismo esquema que metadata.parquet de DiffusionDB.
ROWS = [
    ("a.png", "a cat astronaut, 4k", 1, 11, 50, 7.0, 8, 512, 512, "u1", "2022-08-10 10:00:00+00", 0.1, 0.05),
    ("b.png", "portrait of a dog", 1, 22, 30, 12.0, 1, 512, 768, "u2", "2022-08-11 10:00:00+00", 0.9, 0.2),
    ("c.png", "100%_real cat photo", 2, 33, 50, 7.5, 8, 768, 512, "u1", "2022-08-12 10:00:00+00", 2.0, 0.8),
    ("d.png", None, 2, 44, 20, 3.0, 9, 512, 512, None, None, None, None),
    # El parquet real tiene cfg NaN/inf; no deben romper las estadísticas.
    ("e.png", "broken cfg", 3, 55, 40, float("nan"), 8, 512, 512, "u2", "2022-08-13 10:00:00+00", None, None),
    ("f.png", "inf cfg", 3, 66, 40, float("inf"), 8, 512, 512, "u2", "2022-08-14 10:00:00+00", None, None),
]


@pytest.fixture()
def parquet_path(tmp_path: Path) -> Path:
    path = tmp_path / "metadata.parquet"
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE m (image_name VARCHAR, prompt VARCHAR, part_id USMALLINT, seed UBIGINT, "
        "step USMALLINT, cfg FLOAT, sampler UTINYINT, width USMALLINT, height USMALLINT, "
        "user_name VARCHAR, timestamp TIMESTAMPTZ,image_nsfw FLOAT, prompt_nsfw FLOAT)"
    )
    con.executemany("INSERT INTO m VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", ROWS)
    con.execute(f"COPY m TO '{path.as_posix()}' (FORMAT PARQUET)")
    con.close()
    return path


@pytest.fixture()
def client(parquet_path: Path):
    settings = Settings(data_dir=parquet_path.parent, variant="2m")
    with TestClient(create_app(settings)) as c:
        yield c
