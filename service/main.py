from fastapi import FastAPI, HTTPException

from .config import settings
from .embeddings import embed_images
from .image_io import load_image_from_source
from .model import ModelBundle, load_model_bundle, resolve_device
from .supported_models import load_supported_model_ids
from .schemas import EmbeddingItem, EmbeddingRequest, EmbeddingResponse, EmbeddingVectors, Usage


app = FastAPI(title="imbeddings", version=settings.service_version)


def _get_bundle(model_id: str) -> ModelBundle:
    try:
        return load_model_bundle(model_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/info")
def info() -> dict:
    device = resolve_device()
    return {
        "service": "imbeddings",
        "version": settings.service_version,
        "device": str(device),
        "cuda_memory_fraction": settings.cuda_memory_fraction,
        "max_loaded_models": settings.max_loaded_models,
        "max_batch_size": settings.max_batch_size,
        "max_image_width": settings.max_image_width,
        "max_image_height": settings.max_image_height,
        "max_image_bytes": settings.max_image_bytes,
        "remote_image_request_timeout_seconds": settings.remote_image_request_timeout_seconds,
        "supported_models": load_supported_model_ids(),
    }


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    if not request.model:
        raise HTTPException(status_code=400, detail="model is required")
    inputs = request.input
    if not inputs:
        raise HTTPException(status_code=400, detail="input must not be empty")
    if len(inputs) > settings.max_batch_size:
        raise HTTPException(status_code=400, detail="input exceeds MAX_BATCH_SIZE")

    images = []
    for index, item in enumerate(inputs):
        source = item.image_url or item.image_base64
        try:
            image = load_image_from_source(source)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image at index {index}: {exc}",
            ) from exc
        images.append(image)

    bundle = _get_bundle(request.model)
    cls_vecs, mean_vecs = embed_images(
        images,
        bundle.processor,
        bundle.model,
        bundle.device,
        normalize=True,
    )

    data = []
    for index in range(len(images)):
        data.append(
            EmbeddingItem(
                index=index,
                embeddings=EmbeddingVectors(cls=cls_vecs[index], mean=mean_vecs[index]),
            )
        )

    return EmbeddingResponse(
        data=data,
        model=request.model,
        usage=Usage(input_images=len(images), embedding_dim=len(cls_vecs[0])),
    )
