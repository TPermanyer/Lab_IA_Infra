from pathlib import Path

import pytest

from diffusiondb_api.config import Settings
from diffusiondb_api.main import create_app
from diffusiondb_api.repository import DiffusionDBRepository


def names(resp) -> list[str]:
    return [r["image_name"] for r in resp.json()["data"]]


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_lists_all_prompts_with_pagination_meta(client):
    resp = client.get("/prompts")
    body = resp.json()
    assert resp.status_code == 200
    assert body["success"] is True
    assert body["meta"] == {"total": 6, "limit": 50, "offset": 0}
    assert names(resp) == ["a.png", "b.png", "c.png", "d.png", "e.png", "f.png"]


def test_non_finite_cfg_serializes_as_null(client):
    assert client.get("/prompts/e.png").json()["data"]["cfg"] is None


def test_paginates_with_limit_and_offset(client):
    resp = client.get("/prompts", params={"limit": 2, "offset": 1})
    assert names(resp) == ["b.png", "c.png"]
    assert resp.json()["meta"]["total"] == 6


def test_text_search_is_case_insensitive(client):
    assert names(client.get("/prompts", params={"q": "CAT"})) == ["a.png", "c.png"]


def test_text_search_treats_like_wildcards_literally(client):
    assert names(client.get("/prompts", params={"q": "100%_r"})) == ["c.png"]
    assert names(client.get("/prompts", params={"q": "%"})) == ["c.png"]


def test_filters_combine(client):
    params = {"sampler": 8, "min_step": 40, "max_cfg": 7.2, "width": 512, "height": 512}
    assert names(client.get("/prompts", params=params)) == ["a.png"]


def test_range_and_nsfw_filters(client):
    assert names(client.get("/prompts", params={"max_step": 30})) == ["b.png", "d.png"]
    assert names(client.get("/prompts", params={"min_cfg": 7.5})) == ["b.png", "c.png"]
    assert names(client.get("/prompts", params={"max_image_nsfw": 0.5})) == ["a.png"]
    assert names(client.get("/prompts", params={"max_prompt_nsfw": 0.3})) == ["a.png", "b.png"]


def test_record_includes_sampler_name(client):
    record = client.get("/prompts/a.png").json()["data"]
    assert record["sampler"] == 8
    assert record["sampler_name"] == "k_lms"
    assert record["prompt"] == "a cat astronaut, 4k"


def test_unknown_image_returns_404_envelope(client):
    resp = client.get("/prompts/nope.png")
    assert resp.status_code == 404
    assert resp.json() == {"success": False, "data": None, "error": "Imagen no encontrada: nope.png", "meta": None}


def test_non_finite_cfg_is_null_in_both_cases(client):
    assert client.get("/prompts/f.png").json()["data"]["cfg"] is None
    assert client.get("/prompts/a.png").json()["data"]["cfg"] == 7.0


def test_unknown_route_uses_error_envelope(client):
    resp = client.get("/nope")
    assert resp.status_code == 404
    assert resp.json()["success"] is False
    assert resp.json()["error"] == "Not Found"


@pytest.mark.parametrize(
    "params",
    [{"limit": 0}, {"limit": 501}, {"offset": -1}, {"offset": 100_001}, {"sampler": 10}, {"q": "x" * 201}],
)
def test_rejects_invalid_params_with_error_envelope(client, params):
    resp = client.get("/prompts", params=params)
    body = resp.json()
    assert resp.status_code == 422
    assert body["success"] is False
    assert body["data"] is None
    assert body["error"].startswith("Parámetros inválidos: ")


def test_stats(client):
    stats = client.get("/stats").json()["data"]
    assert stats["total_images"] == 6
    assert stats["unique_prompts"] == 5
    assert stats["unique_users"] == 2
    assert stats["avg_step"] == 38.33
    assert stats["median_cfg"] == 7.25  # mediana de 3, 7, 7.5, 12 (NaN/inf excluidos)
    assert stats["by_sampler"][0] == {"key": "k_lms", "count": 4}
    assert stats["top_resolutions"][0] == {"key": "512x512", "count": 4}
    assert stats["first_timestamp"] == "2022-08-10T10:00:00Z"
    assert stats["last_timestamp"] == "2022-08-14T10:00:00Z"


def test_missing_parquet_fails_fast(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="diffusiondb_api.download"):
        DiffusionDBRepository(tmp_path / "missing.parquet")


def test_app_startup_fails_without_data(tmp_path: Path):
    from fastapi.testclient import TestClient

    app = create_app(Settings(data_dir=tmp_path))
    with pytest.raises(FileNotFoundError):
        with TestClient(app):
            pass
