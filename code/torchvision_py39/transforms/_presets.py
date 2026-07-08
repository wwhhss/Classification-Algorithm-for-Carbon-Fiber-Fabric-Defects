from __future__ import annotations
from enum import Enum
from typing import Any

class InterpolationMode(Enum):
    NEAREST = 0
    BILINEAR = 2
    BICUBIC = 3

class ImageClassification:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
    def __call__(self, img: Any) -> Any:
        return img
    def __repr__(self) -> str:
        return f"ImageClassification(args={self.args}, kwargs={self.kwargs})"
