#!/usr/bin/env python3
"""
clusterer.py — DBSCAN 2D pour regrouper les points LiDAR en objets distincts.

Paramètres :
  k : nombre minimum de voisins pour qu'un point soit core point (DBSCAN minPts)
  D : rayon de voisinage en mètres (DBSCAN eps)
"""
import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points_numpy
from .utils import make_pointcloud2, declare_param


class Clusterer(Node):
    def __init__(self):
        super().__init__("clusterer")
        declare_param(self, "k", 2)
        declare_param(self, "D", 0.15)

        self.pub = self.create_publisher(PointCloud2, "clusters", 10)
        self.sub = self.create_subscription(PointCloud2, "points_filtered", self.callback, 10)

    def callback(self, msg: PointCloud2):
        points = read_points_numpy(msg, ["x", "y", "intensity", "clusterId"])
        if len(points) == 0:
            return
        points[:, 3] = self.dbscan(points[:, :2], eps=self.D, min_pts=self.k)
        self.pub.publish(make_pointcloud2(msg.header, *points.T))

    @staticmethod
    def dbscan(xy: np.ndarray, eps: float, min_pts: int) -> np.ndarray:
        """
        DBSCAN 2D. Retourne un tableau d'entiers : cluster id >= 0, ou -1 (bruit).
        """
        n      = len(xy)
        labels  = np.full(n, -1, dtype=int)
        visited = np.zeros(n, dtype=bool)
        eps2    = eps ** 2
        cid     = 0

        def neighbors(i):
            d2 = ((xy - xy[i]) ** 2).sum(axis=1)
            return np.where(d2 <= eps2)[0]

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            nb = neighbors(i)
            if len(nb) < min_pts:
                continue                       # bruit provisoire
            labels[i] = cid
            seeds = list(nb)
            si = 0
            while si < len(seeds):
                q = seeds[si]; si += 1
                if not visited[q]:
                    visited[q] = True
                    qnb = neighbors(q)
                    if len(qnb) >= min_pts:
                        seeds.extend(qnb[~np.isin(qnb, seeds)])
                if labels[q] == -1:
                    labels[q] = cid
            cid += 1

        return labels.astype(np.float32)


def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(Clusterer())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
