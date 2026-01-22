import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


load_dotenv()

SERVICE_VERION = "0.1.0"


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"Invalid {name} value: {value}") from exc


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"Invalid {name} value: {value}") from exc


def _get_env_optional_float(name: str) -> Optional[float]:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"Invalid {name} value: {value}") from exc


@dataclass(frozen=True)
class Settings:
    device: str
    cuda_memory_fraction: Optional[float]
    max_loaded_models: int
    max_batch_size: int
    max_image_width: int
    max_image_height: int
    max_image_bytes: int
    remote_image_request_timeout_seconds: float
    service_version: str


_max_image_width = _get_env_int("IMBEDDINGS_MAX_IMAGE_WIDTH", 256)
_max_image_height = _get_env_int("IMBEDDINGS_MAX_IMAGE_HEIGHT", 256)

settings = Settings(
    device=os.getenv("IMBEDDINGS_DEVICE", "auto"),
    cuda_memory_fraction=_get_env_optional_float("IMBEDDINGS_CUDA_MEMORY_FRACTION"),
    max_loaded_models=_get_env_int("IMBEDDINGS_MAX_LOADED_MODELS", 1),
    max_batch_size=_get_env_int("IMBEDDINGS_MAX_BATCH_SIZE", 4),
    max_image_width=_max_image_width,
    max_image_height=_max_image_height,
    max_image_bytes=_get_env_int("IMBEDDINGS_MAX_IMAGE_BYTES", 102_400),
    remote_image_request_timeout_seconds=_get_env_float(
        "IMBEDDINGS_REMOTE_IMAGE_REQUEST_TIMEOUT",
        10.0,
    ),
    service_version=SERVICE_VERION,
)

if settings.max_loaded_models < 1:
    raise RuntimeError("IMBEDDINGS_MAX_LOADED_MODELS must be >= 1")

if settings.cuda_memory_fraction is not None and not (
    0.0 < settings.cuda_memory_fraction <= 1.0
):
    raise RuntimeError("IMBEDDINGS_CUDA_MEMORY_FRACTION must be > 0.0 and <= 1.0")
