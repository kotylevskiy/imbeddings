from typing import List, Tuple

import torch
from PIL import Image


@torch.inference_mode()
def embed_images(
    images: List[Image.Image],
    processor,
    model,
    device,
    normalize: bool = True,
) -> Tuple[List[List[float]], List[List[float]]]:
    inputs = processor(images=images, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    outputs = model(**inputs)
    hidden = outputs.last_hidden_state  # (batch, tokens, dim)

    cls_vecs = hidden[:, 0, :]
    mean_vecs = hidden[:, 1:, :].mean(dim=1)

    if normalize:
        cls_vecs = torch.nn.functional.normalize(cls_vecs, p=2, dim=1)
        mean_vecs = torch.nn.functional.normalize(mean_vecs, p=2, dim=1)

    return cls_vecs.cpu().tolist(), mean_vecs.cpu().tolist()
