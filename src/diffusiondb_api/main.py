"""Aplicación FastAPI.

Arranque:
    uv run uvicorn diffusiondb_api.main:app --reload
"""

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from diffusiondb_api.config import Settings, get_settings
from diffusiondb_api.repository import DiffusionDBRepository, PromptFilters
from diffusiondb_api.schemas import ApiResponse, DatasetStats, PageMeta, PromptRecord

DEFAULT_LIMIT = 50
MAX_LIMIT = 500
MAX_QUERY_LENGTH = 200
# Offsets profundos obligan a ordenar millones de filas; para ir más allá, filtrar.
MAX_OFFSET = 100_000


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.repo = DiffusionDBRepository(settings.parquet_path)
        try:
            yield
        finally:
            app.state.repo.close()

    app = FastAPI(
        title="DiffusionDB API",
        description="Consulta de los metadatos (prompts y parámetros) de DiffusionDB.",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Errores con el mismo sobre {success, data, error, meta} que las respuestas OK.
    # Se usa la HTTPException de Starlette para cubrir también los 404/405 de rutas.
    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _error(exc.status_code, str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        detail = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'][1:])}: {e['msg']}" for e in exc.errors()
        )
        return _error(422, f"Parámetros inválidos: {detail}")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/prompts", response_model=ApiResponse[list[PromptRecord]])
    def search_prompts(
        repo: Annotated[DiffusionDBRepository, Depends(get_repo)],
        q: Annotated[str | None, Query(max_length=MAX_QUERY_LENGTH, description="Texto a buscar en el prompt")] = None,
        sampler: Annotated[int | None, Query(ge=1, le=9)] = None,
        min_step: Annotated[int | None, Query(ge=0)] = None,
        max_step: Annotated[int | None, Query(ge=0)] = None,
        min_cfg: Annotated[float | None, Query(ge=0)] = None,
        max_cfg: Annotated[float | None, Query(ge=0)] = None,
        width: Annotated[int | None, Query(gt=0)] = None,
        height: Annotated[int | None, Query(gt=0)] = None,
        max_image_nsfw: Annotated[float | None, Query(ge=0, le=2)] = None,
        max_prompt_nsfw: Annotated[float | None, Query(ge=0, le=1)] = None,
        limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = DEFAULT_LIMIT,
        offset: Annotated[int, Query(ge=0, le=MAX_OFFSET)] = 0,
    ) -> ApiResponse[list[PromptRecord]]:
        filters = PromptFilters(
            q=q, sampler=sampler, min_step=min_step, max_step=max_step,
            min_cfg=min_cfg, max_cfg=max_cfg, width=width, height=height,
            max_image_nsfw=max_image_nsfw, max_prompt_nsfw=max_prompt_nsfw,
        )
        records, total = repo.search(filters, limit=limit, offset=offset)
        return ApiResponse(data=records, meta=PageMeta(total=total, limit=limit, offset=offset))

    @app.get("/prompts/{image_name}", response_model=ApiResponse[PromptRecord])
    def get_prompt(
        image_name: str, repo: Annotated[DiffusionDBRepository, Depends(get_repo)]
    ) -> ApiResponse[PromptRecord]:
        record = repo.get(image_name)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Imagen no encontrada: {image_name}")
        return ApiResponse(data=record)

    @app.get("/stats", response_model=ApiResponse[DatasetStats])
    def stats(repo: Annotated[DiffusionDBRepository, Depends(get_repo)]) -> ApiResponse[DatasetStats]:
        return ApiResponse(data=repo.stats())

    return app


def _error(status_code: int, message: str) -> JSONResponse:
    body = ApiResponse[None](success=False, error=message)
    return JSONResponse(status_code=status_code, content=body.model_dump())


def get_repo(request: Request) -> DiffusionDBRepository:
    return request.app.state.repo


app = create_app()
