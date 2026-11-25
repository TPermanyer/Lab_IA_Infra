# Lab_IA_Infra

## API de datos: DiffusionDB

API REST (FastAPI + DuckDB) para consultar los **metadatos** de
[DiffusionDB](https://huggingface.co/datasets/poloclub/diffusiondb): 2 millones de
prompts de Stable Diffusion con sus parámetros de generación (seed, steps, cfg,
sampler, tamaño), usuario anonimizado, fecha y puntuaciones NSFW.

Solo se descargan los metadatos (`metadata.parquet`, ~195 MB), no las imágenes
(~1.6 TB). DuckDB consulta el parquet directamente, sin cargarlo entero en memoria.

### Requisitos

- [uv](https://docs.astral.sh/uv/)

### Puesta en marcha

```bash
uv sync                                    # instala dependencias
uv run python -m diffusiondb_api.download  # descarga data/metadata.parquet
uv run uvicorn diffusiondb_api.main:app --reload
```

Documentación interactiva en <http://127.0.0.1:8000/docs>.

La descarga reanuda automáticamente si se corta la conexión. Con
`--variant large` se baja DiffusionDB Large (14M filas); para servirla, arranca
la API con `DIFFUSIONDB_VARIANT=large`.

### Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Estado del servicio |
| GET | `/prompts` | Búsqueda paginada con filtros |
| GET | `/prompts/{image_name}` | Metadatos de una imagen |
| GET | `/stats` | Estadísticas globales del dataset |

Filtros de `/prompts`: `q` (texto en el prompt, sin distinguir mayúsculas),
`sampler` (1–9), `min_step`, `max_step`, `min_cfg`, `max_cfg`, `width`, `height`,
`max_image_nsfw`, `max_prompt_nsfw`, `limit` (1–500, por defecto 50) y `offset`
(máximo 100 000; para llegar más lejos, filtra). Los errores (404, 422) usan el
mismo formato, con `success: false` y el mensaje en `error`.

Todas las respuestas usan el mismo formato:

```json
{ "success": true, "data": [...], "error": null, "meta": { "total": 418109, "limit": 50, "offset": 0 } }
```

Ejemplo: `GET /prompts?q=astronaut&max_image_nsfw=0.5&limit=10`

### Notas sobre los datos

- Las fechas se devuelven en UTC.
- `cfg` contiene valores corruptos (NaN, infinito, negativos y hasta 2e28). Por
  eso `/stats` da la **mediana** de cfg y no la media, y los filtros de cfg ignoran
  los valores no finitos.

### Configuración

Variables de entorno (o fichero `.env`):

| Variable | Por defecto | Descripción |
|---|---|---|
| `DIFFUSIONDB_DATA_DIR` | `data` | Carpeta del parquet |
| `DIFFUSIONDB_VARIANT` | `2m` | `2m` o `large` |

### Tests

```bash
uv run pytest
```
