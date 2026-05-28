"""
object_classifier.py
---------------------
Fusion LiDAR (forme + taille + réflexivité) + caméra (couleur HSV)
+ ultrason pour classifier les 9 objets du sujet SY31.

Entrées :
  lidar   : LidarAnalysis (distance, largeur, réflexivité du cluster frontal)
  bbox    : BBoxResult    (cx, cy, largeur, longueur, aspect_ratio) — depuis shaper_bbox
  cylinder: CylinderResult(cx, cy, rayon, residual) — depuis shaper_cylinder
            residual faible → objet circulaire → panneau rond
  polyline: PolylineResult(n_segments, max_angle_deg) — depuis shaper_polyline
            angle ≈ 90° → panneau rectangulaire
  color   : ColorDetection (has_red, has_blue, has_reflective, ratios)
  sonar   : float brut ultrason

Règles de classification (scores [0,1] par objet) :
  panneau_rond          → cercle (cylinder.residual faible) + rouge + taille moyenne
  panneau_rectangulaire → rectangle (polyline.max_angle ≈ 90°) + peu coloré + taille moyenne
  gros_carton_rouge     → rouge, grand (bbox), non réfléchissant
  gros_carton_bleu      → bleu, grand
  petit_carton_rouge    → rouge, petit
  petit_carton_bleu     → bleu, petit
  petit_carton_refl     → réfléchissant, petit
  gros_carton_refl      → réfléchissant, grand
  vitre                 → réfléchissant, grand, LiDAR absent/bruité
"""

from __future__ import annotations
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Optional, Dict

from .lidar_analyzer   import LidarAnalysis
from .detect           import ColorDetection
from .shape_results    import BBoxResult, CylinderResult, PolylineResult


OBJECT_LABELS = [
    "panneau_rond",
    "panneau_rectangulaire",
    "gros_carton_rouge",
    "gros_carton_bleu",
    "petit_carton_rouge",
    "petit_carton_bleu",
    "petit_carton_refl",
    "gros_carton_refl",
    "vitre",
    "inconnu",
]


@dataclass
class Detection:
    label:         str
    score:         float
    confirmed:     bool  = False
    lidar_dist_m:  float = -1.0
    lidar_width_m: float = -1.0
    is_large:      bool  = False
    is_reflective: bool  = False
    color:         str   = "none"
    shape:         str   = "none"     # circle | rectangle | none
    cylinder_residual: float = -1.0   # faible = circulaire
    bbox_aspect:       float = -1.0   # >1.3 = rectangle

    def __str__(self):
        return (f"[{self.label}] {self.score:.2f} "
                f"dist={self.lidar_dist_m:.2f}m color={self.color} shape={self.shape}")


class ObjectClassifier:

    def __init__(
        self,
        confirmation_frames: int   = 3,
        min_score:           float = 0.35,
        size_threshold_m:    float = 0.30,
        sonar_close_raw:     float = 3000.0,
        # Seuils forme
        circle_residual_threshold: float = 0.05,  # résidu < seuil → circulaire
        rect_angle_threshold:      float = 60.0,  # angle > seuil (°) → rectangle
        w_color: float = 0.40,
        w_lidar: float = 0.35,
        w_shape: float = 0.25,
    ):
        self.confirmation_frames         = confirmation_frames
        self.min_score                   = min_score
        self.size_threshold_m            = size_threshold_m
        self.sonar_close_raw             = sonar_close_raw
        self.circle_residual_threshold   = circle_residual_threshold
        self.rect_angle_threshold        = rect_angle_threshold
        self.w_color = w_color
        self.w_lidar = w_lidar
        self.w_shape = w_shape
        self._history: deque[str] = deque(maxlen=confirmation_frames)

    def classify(
        self,
        lidar:    LidarAnalysis,
        color:    ColorDetection,
        bbox:     Optional[BBoxResult]     = None,
        cylinder: Optional[CylinderResult] = None,
        polyline: Optional[PolylineResult] = None,
        sonar_raw: float = 0.0,
    ) -> Optional[Detection]:

        # Pas d'indice du tout → rien
        if (not lidar.object_present
                and not color.has_red
                and not color.has_blue
                and not color.has_reflective):
            self._history.append("none")
            return None

        scores = self._score_all(lidar, color, bbox, cylinder, polyline, sonar_raw)
        best   = max(scores, key=scores.get)
        score  = scores[best]

        if score < self.min_score:
            self._history.append("none")
            return None

        fc = lidar.front_cluster
        det = Detection(
            label             = best,
            score             = score,
            lidar_dist_m      = fc.distance_m      if fc else -1.0,
            lidar_width_m     = fc.width_metric_m  if fc else -1.0,
            is_large          = fc.is_large         if fc else False,
            is_reflective     = fc.is_reflective    if fc else False,
            color             = ("red"        if color.has_red
                                 else "blue"  if color.has_blue
                                 else "reflective" if color.has_reflective
                                 else "none"),
            shape             = self._dominant_shape(cylinder, polyline),
            cylinder_residual = cylinder.residual if cylinder else -1.0,
            bbox_aspect       = bbox.aspect_ratio if bbox else -1.0,
        )

        self._history.append(best)
        det.confirmed = (Counter(self._history).most_common(1)[0] == (best, self.confirmation_frames))
        return det

    # ──────────────────────────────────────────────────────────────────────
    # Scoring
    # ──────────────────────────────────────────────────────────────────────

    def _score_all(self, lidar, color, bbox, cylinder, polyline, sonar_raw) -> Dict[str, float]:
        fc            = lidar.front_cluster
        lidar_present = lidar.object_present
        is_large      = fc.is_large      if fc else False
        is_small      = not is_large
        is_refl_lidar = fc.is_reflective if fc else False
        lidar_absent  = not lidar_present

        sonar_ok = 0 < sonar_raw < self.sonar_close_raw

        # ── Indices de forme ─────────────────────────────────────────────
        is_circle = (
            cylinder is not None
            and cylinder.residual < self.circle_residual_threshold
            and cylinder.residual >= 0
        )
        is_rect = (
            polyline is not None
            and polyline.max_angle_deg > self.rect_angle_threshold
        )
        has_high_aspect = bbox is not None and bbox.aspect_ratio > 1.3

        def fuse(sc, sl, ss):
            return self.w_color * min(sc, 1.0) + self.w_lidar * min(sl, 1.0) + self.w_shape * min(ss, 1.0)

        scores = {}

        # 1. Panneau rond : cercle + rouge
        sc = (0.8 if color.has_red else 0.0)
        sl = (0.7 if lidar_present else 0.0)
        ss = (1.0 if is_circle else 0.2)
        scores["panneau_rond"] = fuse(sc, sl, ss)

        # 2. Panneau rectangulaire : rectangle + peu coloré
        sc = (0.5 if (not color.has_red or color.red_ratio < 0.3) and not color.has_blue else 0.1) \
           + (0.3 if has_high_aspect else 0.0)
        sl = (0.7 if lidar_present else 0.0)
        ss = (1.0 if is_rect else 0.2)
        scores["panneau_rectangulaire"] = fuse(sc, sl, ss)

        # 3. Gros carton rouge
        sc = (0.9 if color.has_red and color.red_ratio > 0.10 else 0.4 if color.has_red else 0.0)
        sl = (0.8 if lidar_present and is_large and not is_refl_lidar else
              0.4 if lidar_present and not is_refl_lidar else 0.0)
        ss = (0.6 if not is_circle else 0.1)
        scores["gros_carton_rouge"] = fuse(sc, sl, ss)

        # 4. Gros carton bleu
        sc = (0.9 if color.has_blue and color.blue_ratio > 0.10 else 0.4 if color.has_blue else 0.0)
        sl = (0.8 if lidar_present and is_large and not is_refl_lidar else
              0.4 if lidar_present and not is_refl_lidar else 0.0)
        ss = (0.6 if not is_circle else 0.1)
        scores["gros_carton_bleu"] = fuse(sc, sl, ss)

        # 5. Petit carton rouge
        sc = (0.9 if color.has_red and color.red_ratio < 0.20 else 0.5 if color.has_red else 0.0)
        sl = (0.8 if lidar_present and is_small and not is_refl_lidar else
              0.4 if lidar_present else 0.0)
        ss = (0.6 if not is_circle else 0.1)
        scores["petit_carton_rouge"] = fuse(sc, sl, ss)

        # 6. Petit carton bleu
        sc = (0.9 if color.has_blue and color.blue_ratio < 0.20 else 0.5 if color.has_blue else 0.0)
        sl = (0.8 if lidar_present and is_small and not is_refl_lidar else
              0.4 if lidar_present else 0.0)
        ss = (0.6 if not is_circle else 0.1)
        scores["petit_carton_bleu"] = fuse(sc, sl, ss)

        # 7. Petit carton réfléchissant
        sc = (0.8 if color.has_reflective and not color.has_red and not color.has_blue else
              0.3 if color.has_reflective else 0.0)
        sl = (0.7 if lidar_present and is_small and is_refl_lidar else
              0.4 if lidar_present and is_small else 0.2 if lidar_present else 0.0)
        ss = 0.5
        scores["petit_carton_refl"] = fuse(sc, sl, ss)

        # 8. Gros carton réfléchissant
        sc = (0.8 if color.has_reflective and not color.has_red and not color.has_blue else
              0.3 if color.has_reflective else 0.0)
        sl = (0.8 if lidar_present and is_large and is_refl_lidar else
              0.4 if lidar_present and is_large else 0.2 if lidar_present else 0.0)
        ss = 0.5
        scores["gros_carton_refl"] = fuse(sc, sl, ss)

        # 9. Vitre : réfléchissante, grande, LiDAR souvent absent
        sc = (0.8 if color.has_reflective else 0.1)
        sl = (0.9 if lidar_absent else
              0.4 if lidar_present and is_refl_lidar and is_large else 0.1)
        ss = 0.5
        scores["vitre"] = fuse(sc, sl, ss)

        return scores

    def _dominant_shape(self, cylinder, polyline) -> str:
        is_circle = (cylinder is not None
                     and cylinder.residual >= 0
                     and cylinder.residual < self.circle_residual_threshold)
        is_rect   = (polyline is not None
                     and polyline.max_angle_deg > self.rect_angle_threshold)
        if is_circle and (not is_rect or cylinder.residual < 0.03):
            return "circle"
        if is_rect:
            return "rectangle"
        return "none"

    def reset(self):
        self._history.clear()
