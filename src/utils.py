from pathlib import Path
from typing import Iterable, Sequence

from .config import DEFAULT_IMAGE_COLUMN, FALLBACK_IMAGE_COLUMNS, MODEL_CONFIGS, MODELS_DIR


def ensure_directory(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_image_column(columns: Sequence[str], preferred: str = DEFAULT_IMAGE_COLUMN) -> str:
    if preferred in columns:
        return preferred

    for column in FALLBACK_IMAGE_COLUMNS:
        if column in columns:
            return column

    raise ValueError(
        f"Could not find an image column. Expected one of: {', '.join(FALLBACK_IMAGE_COLUMNS)}"
    )


def resolve_model_path(model_key: str, model_dir: Path | str = MODELS_DIR) -> Path:
    if model_key not in MODEL_CONFIGS:
        raise KeyError(f"Unsupported model key: {model_key}")
    return Path(model_dir) / MODEL_CONFIGS[model_key].filename


def filter_existing_model_keys(
    model_keys: Iterable[str], model_dir: Path | str = MODELS_DIR
) -> list[str]:
    existing = []
    for model_key in model_keys:
        if resolve_model_path(model_key, model_dir).exists():
            existing.append(model_key)
    return existing
