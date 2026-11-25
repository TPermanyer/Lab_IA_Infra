"""Descarga el parquet de metadatos de DiffusionDB desde Hugging Face.

Uso:
    uv run python -m diffusiondb_api.download            # DiffusionDB 2M (~195 MB)
    uv run python -m diffusiondb_api.download --variant large
"""

import argparse
import sys
import time
from pathlib import Path

import httpx

from diffusiondb_api.config import DATASET_BASE_URL, METADATA_FILES, get_settings

CHUNK_SIZE = 1024 * 1024
TIMEOUT_SECONDS = 60.0
MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 2.0
BACKOFF_MAX_SECONDS = 30.0
PARTIAL_CONTENT = 206
RANGE_NOT_SATISFIABLE = 416
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
PARQUET_MAGIC = b"PAR1"


def build_url(variant: str) -> str:
    if variant not in METADATA_FILES:
        raise ValueError(f"Variante desconocida: {variant!r}. Opciones: {sorted(METADATA_FILES)}")
    return f"{DATASET_BASE_URL}/{METADATA_FILES[variant]}"


def download(url: str, dest: Path, force: bool = False, max_retries: int = MAX_RETRIES) -> Path:
    """Descarga `url` en `dest` de forma atómica (fichero .part + rename).

    Si la conexión se corta, reintenta reanudando desde los bytes ya descargados
    (cabecera Range). Si se agotan los reintentos, borra el fichero parcial.
    """
    if dest.exists() and not force:
        print(f"Ya existe {dest}, se omite la descarga (usa --force para repetir).")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.unlink(missing_ok=True)
    try:
        for attempt in range(1, max_retries + 1):
            try:
                _stream_to(url, tmp)
                break
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                if not _is_retryable(exc) or attempt == max_retries:
                    raise
                delay = min(BACKOFF_BASE_SECONDS * 2 ** (attempt - 1), BACKOFF_MAX_SECONDS)
                print(f"\nFallo transitorio ({exc}); reintento {attempt}/{max_retries - 1} en {delay:.0f}s...")
                time.sleep(delay)
        print()
        _check_parquet(tmp)
        tmp.replace(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return dest


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS
    return True


def _check_parquet(path: Path) -> None:
    """Un parquet válido empieza y termina con los bytes mágicos 'PAR1'."""
    size = path.stat().st_size
    with path.open("rb") as fh:
        head = fh.read(len(PARQUET_MAGIC))
        fh.seek(max(size - len(PARQUET_MAGIC), 0))
        tail = fh.read()
    if size < 2 * len(PARQUET_MAGIC) or head != PARQUET_MAGIC or tail != PARQUET_MAGIC:
        raise ValueError(f"El fichero descargado no es un parquet válido ({size} bytes)")


def _stream_to(url: str, tmp: Path) -> None:
    """Escribe `url` en `tmp`, continuando desde su tamaño actual si ya existe."""
    done = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=TIMEOUT_SECONDS) as resp:
        if done and resp.status_code == RANGE_NOT_SATISFIABLE:
            return  # Ya teníamos todos los bytes; se corta justo al final.
        resp.raise_for_status()
        if done and resp.status_code != PARTIAL_CONTENT:
            done = 0  # El servidor ignoró el Range: empezamos de cero.
        elif done and not resp.headers.get("content-range", "").startswith(f"bytes {done}-"):
            raise httpx.TransportError(f"Content-Range inesperado: {resp.headers.get('content-range')!r}")
        total = done + int(resp.headers.get("content-length", 0))
        with tmp.open("ab" if done else "wb") as fh:
            for chunk in resp.iter_bytes(CHUNK_SIZE):
                fh.write(chunk)
                done += len(chunk)
                _print_progress(done, total)


def _print_progress(done: int, total: int) -> None:
    mb = done / CHUNK_SIZE
    if total:
        print(f"\r{mb:8.1f} MB / {total / CHUNK_SIZE:.1f} MB ({done / total:5.1%})", end="", flush=True)
    else:
        print(f"\r{mb:8.1f} MB", end="", flush=True)


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=sorted(METADATA_FILES), default=settings.variant)
    parser.add_argument("--data-dir", type=Path, default=settings.data_dir)
    parser.add_argument("--force", action="store_true", help="Volver a descargar aunque exista")
    args = parser.parse_args(argv)

    dest = args.data_dir / METADATA_FILES[args.variant]
    try:
        path = download(build_url(args.variant), dest, force=args.force)
    except (httpx.HTTPError, ValueError) as exc:
        print(f"Error descargando DiffusionDB: {exc}", file=sys.stderr)
        return 1
    print(f"Metadatos disponibles en {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
