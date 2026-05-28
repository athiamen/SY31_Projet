#!/usr/bin/env python3
"""
transformer.py — Conversion LaserScan polaire → PointCloud2 cartésien (x, y, intensité).
"""
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
from .utils import make_pointcloud2


class Transformer(Node):
    def __init__(self):
        super().__init__("transformer")
        self.pub = self.create_publisher(PointCloud2, "points", 10)
        self.sub = self.create_subscription(LaserScan, "scan", self.callback, 10)

    def callback(self, msg: LaserScan):
        x, y, intensities = self.scan_to_cartesian(msg)
        self.pub.publish(make_pointcloud2(
            msg.header,
            np.array(x, np.float32),
            np.array(y, np.float32),
            np.array(intensities, np.float32),
        ))

    @staticmethod
    def scan_to_cartesian(msg: LaserScan):
        """Convertit un LaserScan en listes (x, y, intensité)."""
        x, y, intensities = [], [], []
        ranges = np.array(msg.ranges, dtype=np.float32)
        has_i  = len(msg.intensities) == len(ranges)

        for i, theta in enumerate(
            np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
        ):
            if i >= len(ranges):
                break
            r = float(ranges[i])
            # Ignore les points invalides ou trop proches/loin
            if r < msg.range_min or r > msg.range_max or not np.isfinite(r):
                continue
            x.append(r * np.cos(theta))
            y.append(r * np.sin(theta))
            intensities.append(float(msg.intensities[i]) if has_i else 0.0)

        return x, y, intensities


def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(Transformer())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
