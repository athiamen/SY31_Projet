#!/usr/bin/env python3
"""
shaper_polyline.py — Ramer-Douglas-Peucker par cluster.

Un faible nombre de segments avec des angles droits → panneau rectangulaire.
"""
import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points_numpy
from visualization_msgs.msg import MarkerArray
from .utils import make_markers, declare_param
from dataclasses import dataclass
from typing import List


def dist_seg(point, segbeg, segend):
    vec = segend - segbeg
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        return float(np.linalg.norm(point - segbeg))
    vec /= norm
    vecT = np.array([-vec[1], vec[0]])
    return float(abs(np.dot(point - segbeg, vecT)))


@dataclass
class PolylineResult:
    points: list          # liste de [x, y]
    n_segments: int       # nombre de segments après simplification
    max_angle_deg: float  # angle le plus prononcé entre segments consécutifs
                          # proche de 90° → forme rectangulaire


class ShaperPolyline(Node):
    def __init__(self):
        super().__init__("shaper_polyline")
        declare_param(self, "eps", 0.05)
        self.pub = self.create_publisher(MarkerArray, "polylines", 10)
        self.sub = self.create_subscription(PointCloud2, "clusters", self.callback, 10)

    def callback(self, msg: PointCloud2):
        points = read_points_numpy(msg, ["x", "y", "clusterId"])
        results = self.fit(points, self.eps)
        shapes  = [r.points for r in results]
        self.pub.publish(make_markers(msg.header, shapes, self.eps))

    @staticmethod
    def fit(points: np.ndarray, eps: float = 0.05) -> List[PolylineResult]:
        results = []
        for cid in np.unique(points[:, 2]).astype(int):
            if cid < 0:
                continue
            xy = points[points[:, 2].astype(int) == cid, :2]
            # Tri angulaire
            xy = xy[np.argsort(np.arctan2(xy[:, 1], xy[:, 0]))]
            simplified = ShaperPolyline.rdp(xy, eps)
            results.append(PolylineResult(
                points      = simplified,
                n_segments  = max(len(simplified) - 1, 0),
                max_angle_deg = ShaperPolyline._max_angle(simplified),
            ))
        return results

    @staticmethod
    def rdp(xy: np.ndarray, eps: float) -> list:
        """Ramer-Douglas-Peucker récursif."""
        if len(xy) <= 2:
            return xy.tolist()
        dists = np.array([dist_seg(p, xy[0], xy[-1]) for p in xy[1:-1]])
        idx   = int(dists.argmax()) + 1
        if dists[idx - 1] >= eps:
            left  = ShaperPolyline.rdp(xy[:idx + 1], eps)
            right = ShaperPolyline.rdp(xy[idx:],      eps)
            return left[:-1] + right
        return [xy[0].tolist(), xy[-1].tolist()]

    @staticmethod
    def _max_angle(pts: list) -> float:
        """Angle maximal (degrés) entre segments consécutifs d'une polyligne."""
        if len(pts) < 3:
            return 0.0
        arr = np.array(pts)
        angles = []
        for i in range(1, len(arr) - 1):
            v1 = arr[i]   - arr[i-1]
            v2 = arr[i+1] - arr[i]
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
            angles.append(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))
        return float(max(angles)) if angles else 0.0


def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(ShaperPolyline())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
