import base64
from io import BytesIO
import httpx
from PIL import Image

from .config import settings



def _is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _strip_data_uri(value: str) -> str:
    if value.startswith("data:"):
        parts = value.split(",", 1)
        if len(parts) == 2:
            return parts[1]
    return value


def _enforce_limits(image: Image.Image) -> None:
    width, height = image.size
    if width > settings.max_image_width or height > settings.max_image_height:
        raise ValueError("Image dimensions exceed configured limits")


def _load_image_from_bytes(data: bytes) -> Image.Image:
    if len(data) > settings.max_image_bytes:
        raise ValueError("Image size exceeds configured byte limit")
    image = Image.open(BytesIO(data))
    image.load()
    _enforce_limits(image)
    return image.convert("RGB")


def _decode_base64_image(value: str) -> Image.Image:
    raw_value = _strip_data_uri(value).strip()
    try:
        data = base64.b64decode(raw_value, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError("Invalid base64 image data") from exc
    return _load_image_from_bytes(data)


def _fetch_url_bytes(url: str) -> bytes:
    with httpx.Client(
        follow_redirects=True,
        timeout=settings.remote_image_request_timeout_seconds,
    ) as client:
        response = client.get(url)
        response.raise_for_status()
        content_length = response.headers.get("Content-Length")
        if content_length:
            try:
                length = int(content_length)
            except ValueError as exc:
                raise ValueError("Invalid Content-Length header") from exc
            if length > settings.max_image_bytes:
                raise ValueError("Image size exceeds configured byte limit")
        data = response.content
    return data


def load_image_from_source(source: str) -> Image.Image:
    if _is_url(source):
        data = _fetch_url_bytes(source)
        return _load_image_from_bytes(data)
    return _decode_base64_image(source)
