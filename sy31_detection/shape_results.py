from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BBoxResult:
    cx: float
    cy: float
    width: float
    length: float
    aspect_ratio: float
    area: float


@dataclass
class CylinderResult:
    cx: float
    cy: float
    radius: float
    residual: float


@dataclass
class PolylineResult:
    points: list
    n_segments: int
    max_angle_deg: float