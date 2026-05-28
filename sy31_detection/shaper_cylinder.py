#!/usr/bin/env python3
"""
shaper_cylinder.py — Ajustement d'un cylindre (cercle 2D) par cluster.

Un faible résidu d'ajustement indique un objet circulaire → panneau rond.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points_numpy
from visualization_msgs.msg import MarkerArray
from .utils import make_markers
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class CylinderResult:
    cx: float; cy: float; radius: float
    residual: float   # résidu moyen (m) — faible = objet circulaire


def fit_circle(xy: np.ndarray) -> CylinderResult:
    """
    Ajustement d'un cercle par moindres carrés algébriques.
    Résout : 2*cx*x + 2*cy*y + (r²-cx²-cy²) = x²+y²
    """
    if len(xy) < 3:
        cx, cy = float(xy[:, 0].mean()), float(xy[:, 1].mean())
        return CylinderResult(cx=cx, cy=cy, radius=0.05, residual=99.0)

    x, y = xy[:, 0], xy[:, 1]
    A = np.column_stack([2*x, 2*y, np.ones(len(x))])
    b = x**2 + y**2
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = result[0], result[1]
    r = float(np.sqrt(max(result[2] + cx**2 + cy**2, 1e-6)))

    # Résidu : écart moyen entre la distance réelle au centre et le rayon ajusté
    residual = float(np.abs(np.sqrt((x - cx)**2 + (y - cy)**2) - r).mean())

    return CylinderResult(cx=float(cx), cy=float(cy), radius=r, residual=residual)


class ShaperCylinder(Node):
    def __init__(self):
        super().__init__("shaper_cylinder")
        self.pub = self.create_publisher(MarkerArray, "cylinders", 10)
        self.sub = self.create_subscription(PointCloud2, "clusters", self.callback, 10)

    def callback(self, msg: PointCloud2):
        points = read_points_numpy(msg, ["x", "y", "clusterId"])
        results = self.fit(points)
        shapes  = [(r.cx, r.cy, r.radius) for r in results]
        self.pub.publish(make_markers(msg.header, shapes))

    @staticmethod
    def fit(points: np.ndarray) -> List[CylinderResult]:
        results = []
        for cid in np.unique(points[:, 2]).astype(int):
            if cid < 0:
                continue
            xy = points[points[:, 2].astype(int) == cid, :2]
            results.append(fit_circle(xy))
        return results


def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(ShaperCylinder())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
