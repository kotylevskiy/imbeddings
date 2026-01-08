from functools import lru_cache
from pathlib import Path
from typing import List


_SUPPORTED_MODELS_FILENAME = "supported_models.txt"


def _supported_models_path() -> Path:
    return Path(__file__).resolve().parents[1] / _SUPPORTED_MODELS_FILENAME


@lru_cache(maxsize=1)
def load_supported_model_ids() -> List[str]:
    path = _supported_models_path()
    if not path.exists():
        raise RuntimeError(f"Missing {_SUPPORTED_MODELS_FILENAME} at repo root")

    model_ids = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            entry = raw.split("#", 1)[0].strip()
            if not entry:
                continue
            model_ids.append(entry)

    if not model_ids:
        raise RuntimeError("No supported model patterns defined")
    return model_ids


def resolve_model_id(model_id: str) -> str:
    model_ids = load_supported_model_ids()
    if model_id in model_ids:
        return model_id
    raise ValueError(
        "Model is not in supported_models.txt. "
        "Update the file or choose a supported model."
    )
