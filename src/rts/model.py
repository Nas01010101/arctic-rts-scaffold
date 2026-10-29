"""Model registry. `stub` is a two-layer conv net that makes the loop runnable; register a real
segmentation network under a new name and select it with `build_model(name, ...)`."""
from __future__ import annotations

from typing import Callable

from torch import nn

REGISTRY: dict[str, Callable[[int], nn.Module]] = {}


def register(name: str):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


@register("stub")
def _stub(in_channels: int) -> nn.Module:
    return nn.Sequential(nn.Conv2d(in_channels, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 1, 3, padding=1))


def build_model(name: str, in_channels: int) -> nn.Module:
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[name](in_channels)
