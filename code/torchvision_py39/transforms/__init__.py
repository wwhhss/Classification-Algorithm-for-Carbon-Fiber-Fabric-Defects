from __future__ import annotations
from enum import Enum

class InterpolationMode(Enum):
    NEAREST = 0
    BILINEAR = 2
    BICUBIC = 3
