from pathlib import Path

import httpx
import pytest

from diffusiondb_api import download as dl

# Contenido mínimo que pasa la comprobación de bytes mágicos de parquet.
PARQ = b"PAR1" + b"data" + b"PAR1"


class FakeStream:
    def __init__(self, chunks: list[bytes], fail_after: int | None = None, status_code: int = 200,
                 content_range: str | None = None):
        self.chunks = chunks
        self.fail_after = fail_after
        self.status_code = status_code
        self.headers = {"content-length": str(sum(map(len, chunks)))}
        if content_range:
            self.headers["content-range"] = content_range

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", "http://x")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError(f"HTTP {self.status_code}", request=request, response=response)

    def iter_bytes(self, _size):
        for i, chunk in enumerate(self.chunks):
            if self.fail_after is not None and i == self.fail_after:
                raise httpx.ReadError("conexión cortada")
            yield chunk


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(dl.time, "sleep", lambda _s: None)


def serve(monkeypatch, *responses):
    """Hace que httpx.stream devuelva las respuestas en orden y registra las cabeceras enviadas."""
    calls: list[dict] = []
    queue = iter(responses)

    def fake_stream(*_a, headers=None, **_k):
        calls.append(headers)
        return next(queue)

    monkeypatch.setattr(dl.httpx, "stream", fake_stream)
    return calls


def test_build_url():
    assert dl.build_url("2m").endswith("/poloclub/diffusiondb/resolve/main/metadata.parquet")
    assert dl.build_url("large").endswith("/metadata-large.parquet")
    with pytest.raises(ValueError):
        dl.build_url("xl")


def test_download_writes_file(monkeypatch, tmp_path: Path):
    serve(monkeypatch, FakeStream([PARQ[:6], PARQ[6:]]))
    dest = tmp_path / "sub" / "metadata.parquet"
    assert dl.download("http://x", dest) == dest
    assert dest.read_bytes() == PARQ
    assert not dest.with_suffix(".parquet.part").exists()


def test_download_skips_existing_unless_forced(monkeypatch, tmp_path: Path):
    dest = tmp_path / "metadata.parquet"
    dest.write_bytes(b"old")
    serve(monkeypatch, FakeStream([PARQ]))
    dl.download("http://x", dest)
    assert dest.read_bytes() == b"old"
    dl.download("http://x", dest, force=True)
    assert dest.read_bytes() == PARQ


def test_failed_download_leaves_no_partial_file(monkeypatch, tmp_path: Path):
    serve(monkeypatch, *[FakeStream([b"a", b"b"], fail_after=1) for _ in range(dl.MAX_RETRIES)])
    assert dl.main(["--data-dir", str(tmp_path)]) == 1
    assert list(tmp_path.iterdir()) == []


def test_resumes_with_range_after_connection_reset(monkeypatch, tmp_path: Path):
    calls = serve(
        monkeypatch,
        FakeStream([PARQ[:6], PARQ[6:]], fail_after=1),
        FakeStream([PARQ[6:]], status_code=206, content_range=f"bytes 6-{len(PARQ) - 1}/{len(PARQ)}"),
    )
    dest = tmp_path / "metadata.parquet"
    dl.download("http://x", dest)
    assert dest.read_bytes() == PARQ
    assert calls == [{}, {"Range": "bytes=6-"}]


def test_restarts_when_server_ignores_range(monkeypatch, tmp_path: Path):
    serve(monkeypatch, FakeStream([PARQ[:6], PARQ[6:]], fail_after=1), FakeStream([PARQ]))
    dest = tmp_path / "metadata.parquet"
    dl.download("http://x", dest)
    assert dest.read_bytes() == PARQ


def test_unexpected_content_range_is_retried(monkeypatch, tmp_path: Path):
    serve(
        monkeypatch,
        FakeStream([PARQ[:6], PARQ[6:]], fail_after=1),
        FakeStream([b"zz"], status_code=206, content_range="bytes 0-1/12"),
        FakeStream([PARQ[6:]], status_code=206, content_range="bytes 6-11/12"),
    )
    dest = tmp_path / "metadata.parquet"
    dl.download("http://x", dest)
    assert dest.read_bytes() == PARQ


def test_416_after_full_download_means_complete(monkeypatch, tmp_path: Path):
    # Se recibe todo y la conexión se corta al cerrar: el reintento pide un rango vacío.
    serve(monkeypatch, FakeStream([PARQ, b"x"], fail_after=1), FakeStream([], status_code=416))
    dest = tmp_path / "metadata.parquet"
    dl.download("http://x", dest)
    assert dest.read_bytes() == PARQ


def test_retries_transient_http_errors(monkeypatch, tmp_path: Path):
    calls = serve(monkeypatch, FakeStream([], status_code=503), FakeStream([], status_code=429), FakeStream([PARQ]))
    dest = tmp_path / "metadata.parquet"
    dl.download("http://x", dest)
    assert dest.read_bytes() == PARQ
    assert len(calls) == 3


def test_does_not_retry_client_errors(monkeypatch, tmp_path: Path):
    calls = serve(monkeypatch, FakeStream([], status_code=404), FakeStream([PARQ]))
    with pytest.raises(httpx.HTTPStatusError):
        dl.download("http://x", tmp_path / "metadata.parquet")
    assert len(calls) == 1


def test_rejects_non_parquet_content(monkeypatch, tmp_path: Path, capsys):
    serve(monkeypatch, FakeStream([b"<html>error</html>"]))
    assert dl.main(["--data-dir", str(tmp_path)]) == 1
    assert "no es un parquet" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_main_success(monkeypatch, tmp_path: Path, capsys):
    serve(monkeypatch, FakeStream([PARQ]))
    assert dl.main(["--data-dir", str(tmp_path), "--variant", "large"]) == 0
    assert (tmp_path / "metadata-large.parquet").read_bytes() == PARQ
    assert "Metadatos disponibles" in capsys.readouterr().out
