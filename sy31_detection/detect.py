#!/usr/bin/env python3
"""
detect.py — Détection de couleur HSV dans l'image caméra.

Fournit detect_colors() utilisé directement par detection_node pour extraire :
  - présence et ratio de pixels rouges, bleus, réfléchissants
  - bounding box de la zone colorée dominante (pour estimer la taille dans l'image)
"""
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    from cv_bridge import CvBridge, CvBridgeError
    HAS_BRIDGE = True
except ImportError:
    HAS_BRIDGE = False


@dataclass
class ColorDetection:
    has_red:        bool  = False
    has_blue:       bool  = False
    has_reflective: bool  = False
    red_ratio:        float = 0.0
    blue_ratio:       float = 0.0
    reflective_ratio: float = 0.0
    red_bbox:        Optional[Tuple[int,int,int,int]] = None
    blue_bbox:       Optional[Tuple[int,int,int,int]] = None
    reflective_bbox: Optional[Tuple[int,int,int,int]] = None


# Seuils HSV
_RED_LO1 = np.array([0,   80, 60],  dtype=np.uint8)
_RED_HI1 = np.array([10, 255, 255], dtype=np.uint8)
_RED_LO2 = np.array([170, 80, 60],  dtype=np.uint8)
_RED_HI2 = np.array([180,255, 255], dtype=np.uint8)
_BLU_LO  = np.array([100, 80, 60],  dtype=np.uint8)
_BLU_HI  = np.array([130,255, 255], dtype=np.uint8)
_REF_LO  = np.array([0,   0, 180],  dtype=np.uint8)
_REF_HI  = np.array([180, 40, 255], dtype=np.uint8)
_MIN_RATIO = 0.03
_KERNEL    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def detect_colors(bgr: np.ndarray) -> ColorDetection:
    """
    Analyse une image BGR et retourne les couleurs détectées.
    C'est la fonction principale appelée par detection_node.
    """
    if bgr is None or bgr.size == 0:
        return ColorDetection()

    hsv   = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    total = bgr.shape[0] * bgr.shape[1]

    # Masques
    red_mask = cv2.morphologyEx(
        cv2.bitwise_or(cv2.inRange(hsv, _RED_LO1, _RED_HI1),
                       cv2.inRange(hsv, _RED_LO2, _RED_HI2)),
        cv2.MORPH_CLOSE, _KERNEL)
    blu_mask = cv2.morphologyEx(
        cv2.inRange(hsv, _BLU_LO, _BLU_HI), cv2.MORPH_CLOSE, _KERNEL)
    ref_mask = cv2.morphologyEx(
        cv2.inRange(hsv, _REF_LO, _REF_HI), cv2.MORPH_CLOSE, _KERNEL)

    red_r = float(red_mask.sum()) / (255.0 * total)
    blu_r = float(blu_mask.sum()) / (255.0 * total)
    ref_r = float(ref_mask.sum()) / (255.0 * total)

    return ColorDetection(
        has_red         = red_r >= _MIN_RATIO,
        has_blue        = blu_r >= _MIN_RATIO,
        has_reflective  = ref_r >= _MIN_RATIO,
        red_ratio         = red_r,
        blue_ratio        = blu_r,
        reflective_ratio  = ref_r,
        red_bbox        = _largest_bbox(red_mask) if red_r >= _MIN_RATIO else None,
        blue_bbox       = _largest_bbox(blu_mask) if blu_r >= _MIN_RATIO else None,
        reflective_bbox = _largest_bbox(ref_mask) if ref_r >= _MIN_RATIO else None,
    )


def draw_detections(bgr: np.ndarray, det: ColorDetection) -> np.ndarray:
    """Dessine les bounding boxes colorées sur une copie de l'image."""
    out = bgr.copy()
    for bbox, color, label in [
        (det.red_bbox,        (0,0,255),     "Rouge"),
        (det.blue_bbox,       (255,0,0),     "Bleu"),
        (det.reflective_bbox, (200,200,200), "Reflechissant"),
    ]:
        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(out, (x,y), (x+w,y+h), color, 2)
            cv2.putText(out, label, (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def _largest_bbox(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return cv2.boundingRect(max(cnts, key=cv2.contourArea))


# ── Nœud ROS2 autonome (optionnel) ───────────────────────────────────────────
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image

    class Detector(Node):
        def __init__(self):
            super().__init__("detector")
            if not HAS_BRIDGE:
                self.get_logger().error("cv_bridge manquant"); return
            self.bridge = CvBridge()
            self.pub = self.create_publisher(Image, "detections", 10)
            self.sub = self.create_subscription(
                Image, "turtlecam/image_rect", self.callback, 10)

        def callback(self, msg):
            try:
                img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            except CvBridgeError as e:
                self.get_logger().warn(str(e)); return
            det    = detect_colors(img)
            result = draw_detections(img, det)
            try:
                self.pub.publish(self.bridge.cv2_to_imgmsg(result, "bgr8"))
            except CvBridgeError as e:
                self.get_logger().warn(str(e))

    def main(args=None):
        rclpy.init(args=args)
        try:
            rclpy.spin(Detector())
        except KeyboardInterrupt:
            pass

except ImportError:
    pass

if __name__ == "__main__":
    main()
