#!/usr/bin/env python3
"""
detection_node.py — Nœud ROS2 principal de détection d'objets.

Pipeline par scan LiDAR :
  /scan  → Transformer (polaire→XY)
         → IntensityFilter (filtre réflexivité)
         → Clusterer DBSCAN (objets séparés)
         → ShaperBBox    (largeur/longueur → taille)
         → ShaperCylinder(résidu cercle   → forme ronde)
         → ShaperPolyline(angle max       → forme rectangulaire)
         → LidarAnalyzer (distance, réflexivité)

Pipeline par image caméra :
  /turtlecam/image_raw/compressed → detect_colors() (rouge/bleu/réfléchissant)

Fusion → ObjectClassifier → label + score + image annotée

Publications :
  /object_detection/label   std_msgs/String
  /object_detection/score   std_msgs/Float32
  /object_detection/image   sensor_msgs/CompressedImage
  /bboxes   /cylinders   /polylines   visualization_msgs/MarkerArray  (RViz)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import numpy as np
import cv2

from sensor_msgs.msg import CompressedImage, LaserScan, PointCloud2
from std_msgs.msg import String, Float32
from visualization_msgs.msg import MarkerArray

try:
    from turtlebot3_msgs.msg import SensorState
    HAS_TB3 = True
except ImportError:
    HAS_TB3 = False

from .lidar_analyzer   import LidarAnalyzer, LidarAnalysis
from .detect           import detect_colors, draw_detections, ColorDetection
from .object_classifier import ObjectClassifier
from .transformer      import Transformer
from .intensity_filter import IntensityFilter
from .clusterer        import Clusterer
from .shaper_bbox      import ShaperBBox
from .shaper_cylinder  import ShaperCylinder
from .shaper_polyline  import ShaperPolyline
from .utils            import make_pointcloud2, make_markers


class DetectionNode(Node):

    def __init__(self):
        super().__init__("sy31_detection_node")

        # ── Paramètres ────────────────────────────────────────────────────
        self.declare_parameter("topic_image",              "/turtlecam/image_raw/compressed")
        self.declare_parameter("topic_scan",               "/scan")
        self.declare_parameter("topic_sensor",             "/sensor_state")
        self.declare_parameter("front_angle_deg",           30.0)
        self.declare_parameter("max_dist_m",                 2.5)
        self.declare_parameter("refl_threshold",             0.6)
        self.declare_parameter("sonar_close_raw",         3000.0)
        self.declare_parameter("confirmation_frames",        3)
        self.declare_parameter("min_score",                  0.35)
        self.declare_parameter("publish_annotated_image",    True)
        self.declare_parameter("intensity_threshold",        0.0)
        self.declare_parameter("dbscan_k",                   2)
        self.declare_parameter("dbscan_D",                   0.15)
        self.declare_parameter("rdp_eps",                    0.05)
        self.declare_parameter("circle_residual_threshold",  0.05)
        self.declare_parameter("rect_angle_threshold",      60.0)

        p = self.get_parameter

        # ── Composants ───────────────────────────────────────────────────
        self._lidar_analyzer = LidarAnalyzer(
            front_angle_deg = p("front_angle_deg").value,
            max_dist_m      = p("max_dist_m").value,
            refl_threshold  = p("refl_threshold").value,
        )
        self._classifier = ObjectClassifier(
            confirmation_frames       = p("confirmation_frames").value,
            min_score                 = p("min_score").value,
            sonar_close_raw           = p("sonar_close_raw").value,
            circle_residual_threshold = p("circle_residual_threshold").value,
            rect_angle_threshold      = p("rect_angle_threshold").value,
        )

        # Paramètres pipeline LiDAR
        self._intensity_thr = p("intensity_threshold").value
        self._dbscan_k      = p("dbscan_k").value
        self._dbscan_D      = p("dbscan_D").value
        self._rdp_eps       = p("rdp_eps").value

        # ── État courant ─────────────────────────────────────────────────
        self._last_lidar  = LidarAnalysis()
        self._last_color  = ColorDetection()
        self._last_sonar  = 0.0

        # Résultats de forme du dernier scan
        self._last_bbox     = None
        self._last_cylinder = None
        self._last_polyline = None

        # ── QoS best-effort pour le LiDAR ────────────────────────────────
        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=5)

        # ── Souscriptions ────────────────────────────────────────────────
        self.create_subscription(CompressedImage,
            p("topic_image").value, self._cb_image, 10)
        self.create_subscription(LaserScan,
            p("topic_scan").value,  self._cb_scan, qos_be)
        if HAS_TB3:
            self.create_subscription(SensorState,
                p("topic_sensor").value, self._cb_sensor, 10)
        else:
            self.get_logger().warn("turtlebot3_msgs absent – ultrason désactivé.")

        # ── Publications ─────────────────────────────────────────────────
        self._pub_label    = self.create_publisher(String,          "/object_detection/label", 10)
        self._pub_score    = self.create_publisher(Float32,         "/object_detection/score", 10)
        self._pub_image    = self.create_publisher(CompressedImage, "/object_detection/image", 10)
        self._pub_bboxes   = self.create_publisher(MarkerArray,     "/bboxes",      10)
        self._pub_cyls     = self.create_publisher(MarkerArray,     "/cylinders",   10)
        self._pub_polys    = self.create_publisher(MarkerArray,     "/polylines",   10)

        self.get_logger().info("DetectionNode démarré.")

    # ══════════════════════════════════════════════════════════════════════
    # Callbacks
    # ══════════════════════════════════════════════════════════════════════

    def _cb_image(self, msg: CompressedImage):
        data = bytes(msg.data)
        img  = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        self._last_color = detect_colors(img)
        self._classify(img)

    def _cb_scan(self, msg: LaserScan):
        # 1. LidarAnalyzer (distance + réflexivité)
        self._last_lidar = self._lidar_analyzer.analyze(
            ranges          = msg.ranges,
            intensities     = msg.intensities if len(msg.intensities) > 0 else None,
            angle_min       = msg.angle_min,
            angle_increment = msg.angle_increment,
        )

        # 2. Transformer → XY
        x, y, intensities = Transformer.scan_to_cartesian(msg)
        if not x:
            return
        xy_all = np.column_stack([x, y, intensities])

        # 3. IntensityFilter
        xy_filt = IntensityFilter.filter(xy_all, self._intensity_thr)
        if len(xy_filt) == 0:
            return

        # 4. Clusterer DBSCAN
        cluster_ids = Clusterer.dbscan(xy_filt[:, :2],
                                       eps=self._dbscan_D,
                                       min_pts=self._dbscan_k)
        pts_with_id = np.column_stack([xy_filt[:, :2],
                                       xy_filt[:, 2],
                                       cluster_ids])

        # 5. Shapers → résultats utilisés pour la classification
        bbox_results = ShaperBBox.fit(pts_with_id)
        cyl_results  = ShaperCylinder.fit(pts_with_id)
        poly_results = ShaperPolyline.fit(pts_with_id, self._rdp_eps)

        # Garde le résultat du cluster frontal (le plus proche de 0°)
        self._last_bbox     = bbox_results[0]     if bbox_results     else None
        self._last_cylinder = cyl_results[0]      if cyl_results      else None
        self._last_polyline = poly_results[0]      if poly_results     else None

        # Publication RViz
        self._pub_bboxes.publish(make_markers(msg.header,
            [(r.cx, r.cy, r.width, r.length) for r in bbox_results]))
        self._pub_cyls.publish(make_markers(msg.header,
            [(r.cx, r.cy, r.radius) for r in cyl_results]))
        self._pub_polys.publish(make_markers(msg.header,
            [r.points for r in poly_results], self._rdp_eps))

    def _cb_sensor(self, msg):
        self._last_sonar = float(msg.sonar)

    # ══════════════════════════════════════════════════════════════════════
    # Classification
    # ══════════════════════════════════════════════════════════════════════

    def _classify(self, bgr: np.ndarray):
        det = self._classifier.classify(
            lidar    = self._last_lidar,
            color    = self._last_color,
            bbox     = self._last_bbox,
            cylinder = self._last_cylinder,
            polyline = self._last_polyline,
            sonar_raw= self._last_sonar,
        )
        if det is None:
            return

        lbl = String();  lbl.data  = det.label
        scr = Float32(); scr.data  = float(det.score)
        self._pub_label.publish(lbl)
        self._pub_score.publish(scr)

        if self.get_parameter("publish_annotated_image").value:
            annotated = draw_detections(bgr, self._last_color)
            color_ov  = (0, 255, 0) if det.confirmed else (0, 200, 255)
            cv2.putText(annotated, f"{det.label} ({det.score*100:.0f}%)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_ov, 2)
            cv2.putText(annotated,
                        f"dist={det.lidar_dist_m:.2f}m  {det.color}  {det.shape}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_ov, 2)
            _, enc = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            out = CompressedImage()
            out.header.stamp = self.get_clock().now().to_msg()
            out.format = "jpeg"; out.data = enc.tobytes()
            self._pub_image.publish(out)

        if det.confirmed:
            self.get_logger().info(
                f"[CONFIRMÉ] {det.label} | score={det.score:.2f} "
                f"shape={det.shape} color={det.color} "
                f"dist={det.lidar_dist_m:.2f}m "
                f"cylinder_residual={det.cylinder_residual:.3f} "
                f"bbox_aspect={det.bbox_aspect:.2f}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
