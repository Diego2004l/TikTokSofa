"""Tier 1 — EfficientNet-B0 CNN backbone (spec section 4, Tier 1).

ImageNet-pretrained via `timm` (Apache 2.0), fine-tuned end-to-end on compound-augmented data.
Outputs a calibrated probability via sigmoid, not just a hard label.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.transforms as T

try:
    import timm
except ImportError as exc:  # pragma: no cover
    raise ImportError("timm is required for src/model/model.py") from exc

INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform() -> T.Compose:
    return T.Compose(
        [
            T.Resize((INPUT_SIZE, INPUT_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class Tier1CNN(nn.Module):
    """EfficientNet-B0 with a single-logit head. `predict_proba` applies sigmoid so the
    output is directly usable as Tier 1's calibrated score in the fusion step."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(-1)  # logits, shape (B,)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


def load_checkpoint(path: str, device: str = "cpu") -> Tier1CNN:
    model = Tier1CNN(pretrained=False)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model
