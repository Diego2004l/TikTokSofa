"""Tier 3 — frozen CLIP ViT-B/32 semantic embeddings (spec section 4, Tier 3).

Deliberately CLIP ViT-B/32 alone (not paired with DINOv2 — see README "Related Work" /
spec section 7 for why that specific combination is avoided). Embeddings are frozen; only the
logistic-regression probe in `train_probe.py` is trained.
"""

from __future__ import annotations

import torch

try:
    import open_clip
except ImportError as exc:  # pragma: no cover
    raise ImportError("open_clip_torch is required for src/semantic/embed.py") from exc

MODEL_NAME = "ViT-B-32-quickgelu"  # matches the activation function the 'openai' weights were trained with
PRETRAINED = "openai"


class ClipEmbedder:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
        self.model.eval().to(device)
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def embed_image(self, img) -> torch.Tensor:
        """`img` is a PIL.Image. Returns an L2-normalized embedding vector."""
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        features = self.model.encode_image(tensor)
        return (features / features.norm(dim=-1, keepdim=True)).squeeze(0).cpu()

    @torch.no_grad()
    def embed_batch(self, imgs: list) -> torch.Tensor:
        tensors = torch.stack([self.preprocess(img) for img in imgs]).to(self.device)
        features = self.model.encode_image(tensors)
        return (features / features.norm(dim=-1, keepdim=True)).cpu()
