#!/usr/bin/env python3
"""
shaper_bbox.py — Bounding box (cx, cy, largeur, longueur) par cluster.

Utilisé pour la détection : la largeur/longueur discrimine petit/grand objet
et aide à distinguer les formes rectangulaires.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points_numpy
from visualization_msgs.msg import MarkerArray
from .utils import make_markers
from dataclasses import dataclass
from typing import List


@dataclass
class BBoxResult:
    cx: float; cy: float
    width: float; length: float
    aspect_ratio: float   # max(w,l)/min(w,l) → >1.3 suggère rectangle
    area: float


class ShaperBBox(Node):
    def __init__(self):
        super().__init__("shaper_bbox")
        self.pub = self.create_publisher(MarkerArray, "bboxes", 10)
        self.sub = self.create_subscription(PointCloud2, "clusters", self.callback, 10)

    def callback(self, msg: PointCloud2):
        points = read_points_numpy(msg, ["x", "y", "clusterId"])
        results = self.fit(points)
        shapes  = [(r.cx, r.cy, r.width, r.length) for r in results]
        self.pub.publish(make_markers(msg.header, shapes))

    @staticmethod
    def fit(points: np.ndarray) -> List[BBoxResult]:
        """Calcule une BBox par cluster. points shape = (N, 3) : x, y, clusterId."""
        results = []
        for cid in np.unique(points[:, 2]).astype(int):
            if cid < 0:
                continue
            xy = points[points[:, 2].astype(int) == cid, :2]
            xmin, ymin = xy.min(axis=0)
            xmax, ymax = xy.max(axis=0)
            w = float(max(xmax - xmin, 0.05))
            l = float(max(ymax - ymin, 0.05))
            results.append(BBoxResult(
                cx           = float((xmin + xmax) / 2),
                cy           = float((ymin + ymax) / 2),
                width        = w,
                length       = l,
                aspect_ratio = max(w, l) / min(w, l),
                area         = w * l,
            ))
        return results


def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(ShaperBBox())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
