#!/usr/bin/env python3
"""
intensity_filter.py — Filtre les points LiDAR par intensité.
Permet d'isoler les surfaces réfléchissantes (vitre, carton argenté).
"""
import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points_numpy
from .utils import make_pointcloud2, declare_param


class IntensityFilter(Node):
    def __init__(self):
        super().__init__("intensity_filter")
        # Seuil : garder uniquement points avec intensité >= threshold
        # 0.0 = tout passe ; monter pour n'isoler que le réfléchissant
        declare_param(self, "intensity_threshold", 0.0)

        self.pub = self.create_publisher(PointCloud2, "points_filtered", 10)
        self.sub = self.create_subscription(PointCloud2, "points", self.callback, 10)

    def callback(self, msg: PointCloud2):
        points = read_points_numpy(msg, ["x", "y", "intensity"])
        if len(points) == 0:
            return
        points_filt = self.filter(points, self.intensity_threshold)
        if len(points_filt) == 0:
            return
        self.pub.publish(make_pointcloud2(
            msg.header, points_filt[:, 0], points_filt[:, 1], points_filt[:, 2]
        ))

    @staticmethod
    def filter(points: np.ndarray, threshold: float) -> np.ndarray:
        """Retourne les points dont l'intensité (colonne 2) >= threshold."""
        return points[points[:, 2] >= threshold]


def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(IntensityFilter())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
