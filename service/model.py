import os
from dataclasses import dataclass
from functools import lru_cache

import torch
from transformers import AutoImageProcessor, AutoModel

from .config import settings
from .supported_models import resolve_model_id


@dataclass(frozen=True)
class ModelBundle:
    processor: AutoImageProcessor
    model: AutoModel
    device: torch.device


def _get_hf_token() -> str:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "Missing HF_TOKEN. "
            "Set it in the environment or a .env file."
        )
    return token


def resolve_device() -> torch.device:
    device_name = settings.device.lower()
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name in {"cpu", "cuda"}:
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("IMBEDDINGS_DEVICE is set to cuda but no CUDA device is available")
        return torch.device(device_name)
    raise RuntimeError("IMBEDDINGS_DEVICE must be one of: auto, cpu, cuda")


@lru_cache(maxsize=settings.max_loaded_models)
def _load_model_bundle(resolved_model_id: str) -> ModelBundle:
    token = _get_hf_token()
    device = resolve_device()

    processor = AutoImageProcessor.from_pretrained(
        resolved_model_id,
        token=token,
    )
    model = AutoModel.from_pretrained(
        resolved_model_id,
        token=token,
    )

    model.to(device)
    model.eval()
    return ModelBundle(processor=processor, model=model, device=device)


def load_model_bundle(model_id: str) -> ModelBundle:
    resolved_model_id = resolve_model_id(model_id)
    return _load_model_bundle(resolved_model_id)
