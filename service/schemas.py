from typing import List, Optional, Literal

from pydantic import BaseModel, Field, model_validator


class EmbeddingInputItem(BaseModel):
    type: Literal["image"] = Field("image", description="Input type")
    image_base64: Optional[str] = Field(
        None,
        description="Base64 image data or data URI",
    )
    image_url: Optional[str] = Field(
        None,
        description="HTTP(S) URL to an image",
    )

    @model_validator(mode="after")
    def _ensure_single_source(self):
        if bool(self.image_base64) == bool(self.image_url):
            raise ValueError("Provide exactly one of image_base64 or image_url")
        return self


class EmbeddingRequest(BaseModel):
    input: List[EmbeddingInputItem] = Field(..., description="List of image inputs")
    model: str = Field(..., description="Model ID", min_length=1)


class EmbeddingVectors(BaseModel):
    cls: List[float]
    mean: List[float]


class EmbeddingItem(BaseModel):
    object: str = "embedding"
    index: int
    embeddings: EmbeddingVectors


class Usage(BaseModel):
    input_images: int
    embedding_dim: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingItem]
    model: str
    usage: Usage
