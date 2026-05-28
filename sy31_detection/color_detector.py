"""
color_detector.py
-----------------
Détection de couleur et de forme dans une image BGR (OpenCV).

Couleurs gérées : rouge, bleu, réfléchissant (argenté / haute brillance).
Formes gérées   : cercle (panneau rond), rectangle (panneau rectangulaire).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np
import cv2


@dataclass
class ColorResult:
    """Résultat de l'analyse couleur d'une image."""
    has_red:        bool  = False
    has_blue:       bool  = False
    has_reflective: bool  = False

    red_ratio:        float = 0.0   # fraction de pixels rouges [0-1]
    blue_ratio:       float = 0.0
    reflective_ratio: float = 0.0

    # Bounding box de la zone colorée dominante (x, y, w, h) ou None
    red_bbox:        Optional[Tuple[int,int,int,int]] = None
    blue_bbox:       Optional[Tuple[int,int,int,int]] = None
    reflective_bbox: Optional[Tuple[int,int,int,int]] = None


@dataclass
class ShapeResult:
    """Résultat de la détection de forme."""
    has_circle:    bool = False
    has_rectangle: bool = False
    circle_score:  float = 0.0   # confiance [0-1]
    rect_score:    float = 0.0


@dataclass
class ImageAnalysis:
    """Résultat combiné couleur + forme."""
    color: ColorResult = field(default_factory=ColorResult)
    shape: ShapeResult = field(default_factory=ShapeResult)
    image_size: Tuple[int,int] = (0, 0)   # (height, width)


class ColorDetector:
    """
    Détecte les couleurs et formes dans une image BGR.

    Parameters
    ----------
    red_h_range1    : (lo, hi) plage H pour rouge bas [0–10]
    red_h_range2    : (lo, hi) plage H pour rouge haut [170–180]
    red_sv_min      : (s_min, v_min) seuils S et V pour rouge
    blue_h_range    : (lo, hi) plage H pour bleu [100–130]
    blue_sv_min     : (s_min, v_min)
    refl_sv         : (s_max, v_min) seuil réfléchissant (faible S, haute V)
    min_color_ratio : ratio minimal pour valider une couleur
    canny_low/high  : seuils Canny pour Hough
    """

    def __init__(
        self,
        red_h_range1: Tuple[int,int]  = (0, 10),
        red_h_range2: Tuple[int,int]  = (170, 180),
        red_sv_min:   Tuple[int,int]  = (80, 60),
        blue_h_range: Tuple[int,int]  = (100, 130),
        blue_sv_min:  Tuple[int,int]  = (80, 60),
        refl_sv:      Tuple[int,int]  = (40, 180),  # (s_max, v_min)
        min_color_ratio: float        = 0.03,
        canny_low:  int = 50,
        canny_high: int = 150,
        hough_dp:   float = 1.2,
        hough_min_dist: int   = 50,
        hough_param1:   float = 80,
        hough_param2:   float = 30,
        hough_min_r:    int   = 20,
        hough_max_r:    int   = 120,
        rect_aspect_min: float = 1.3,
        rect_aspect_max: float = 4.0,
        min_contour_area: float = 1000,
    ):
        self.red_h_range1    = red_h_range1
        self.red_h_range2    = red_h_range2
        self.red_sv_min      = red_sv_min
        self.blue_h_range    = blue_h_range
        self.blue_sv_min     = blue_sv_min
        self.refl_sv         = refl_sv
        self.min_color_ratio = min_color_ratio

        self.canny_low       = canny_low
        self.canny_high      = canny_high
        self.hough_dp        = hough_dp
        self.hough_min_dist  = hough_min_dist
        self.hough_param1    = hough_param1
        self.hough_param2    = hough_param2
        self.hough_min_r     = hough_min_r
        self.hough_max_r     = hough_max_r
        self.rect_aspect_min = rect_aspect_min
        self.rect_aspect_max = rect_aspect_max
        self.min_contour_area = min_contour_area

    # ------------------------------------------------------------------
    # Interface principale
    # ------------------------------------------------------------------

    def analyze(self, bgr_image: np.ndarray) -> ImageAnalysis:
        """
        Analyse une image BGR et retourne couleurs + formes détectées.

        Parameters
        ----------
        bgr_image : image BGR (numpy array H×W×3, uint8)
        """
        if bgr_image is None or bgr_image.size == 0:
            return ImageAnalysis()

        h, w = bgr_image.shape[:2]
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        total_px = h * w

        color = self._detect_colors(hsv, total_px)
        shape = self._detect_shapes(bgr_image, color)

        return ImageAnalysis(color=color, shape=shape, image_size=(h, w))

    # ------------------------------------------------------------------
    # Détection couleur
    # ------------------------------------------------------------------

    def _detect_colors(self, hsv: np.ndarray, total_px: int) -> ColorResult:
        s_min_r, v_min_r = self.red_sv_min
        s_min_b, v_min_b = self.blue_sv_min
        s_max_f, v_min_f = self.refl_sv

        # Rouge
        lo1 = (self.red_h_range1[0], s_min_r, v_min_r)
        hi1 = (self.red_h_range1[1], 255,     255)
        lo2 = (self.red_h_range2[0], s_min_r, v_min_r)
        hi2 = (self.red_h_range2[1], 255,     255)
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, lo1, hi1),
            cv2.inRange(hsv, lo2, hi2),
        )
        red_ratio = float(red_mask.sum()) / (255.0 * total_px)

        # Bleu
        blue_mask = cv2.inRange(
            hsv,
            (self.blue_h_range[0], s_min_b, v_min_b),
            (self.blue_h_range[1], 255,     255),
        )
        blue_ratio = float(blue_mask.sum()) / (255.0 * total_px)

        # Réfléchissant : faible saturation + haute valeur
        refl_mask = cv2.inRange(
            hsv,
            (0,        0,        v_min_f),
            (180,      s_max_f,  255),
        )
        refl_ratio = float(refl_mask.sum()) / (255.0 * total_px)

        return ColorResult(
            has_red        = red_ratio  >= self.min_color_ratio,
            has_blue       = blue_ratio >= self.min_color_ratio,
            has_reflective = refl_ratio >= self.min_color_ratio,
            red_ratio        = red_ratio,
            blue_ratio       = blue_ratio,
            reflective_ratio = refl_ratio,
            red_bbox        = self._largest_bbox(red_mask)  if red_ratio  >= self.min_color_ratio else None,
            blue_bbox       = self._largest_bbox(blue_mask) if blue_ratio >= self.min_color_ratio else None,
            reflective_bbox = self._largest_bbox(refl_mask) if refl_ratio >= self.min_color_ratio else None,
        )

    @staticmethod
    def _largest_bbox(mask: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
        """Retourne la bounding box du plus grand contour dans le masque."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        return cv2.boundingRect(largest)

    # ------------------------------------------------------------------
    # Détection forme
    # ------------------------------------------------------------------

    def _detect_shapes(self, bgr: np.ndarray, color: ColorResult) -> ShapeResult:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        circle_score = self._detect_circle(blurred, color)
        rect_score   = self._detect_rectangle(bgr, color)

        return ShapeResult(
            has_circle    = circle_score > 0.5,
            has_rectangle = rect_score   > 0.5,
            circle_score  = circle_score,
            rect_score    = rect_score,
        )

    def _detect_circle(self, blurred_gray: np.ndarray, color: ColorResult) -> float:
        """
        Détecte un cercle via Hough. Retourne un score [0-1].
        Un panneau rond est rouge → bonus si rouge détecté.
        """
        circles = cv2.HoughCircles(
            blurred_gray,
            cv2.HOUGH_GRADIENT,
            dp=self.hough_dp,
            minDist=self.hough_min_dist,
            param1=self.hough_param1,
            param2=self.hough_param2,
            minRadius=self.hough_min_r,
            maxRadius=self.hough_max_r,
        )
        if circles is None:
            return 0.0

        # Score de base proportionnel au nombre de cercles trouvés (max 1)
        base_score = min(1.0, len(circles[0]) * 0.5)

        # Bonus si couleur rouge présente (panneau de signalisation)
        if color.has_red and color.red_ratio > 0.05:
            base_score = min(1.0, base_score + 0.3)

        return base_score

    def _detect_rectangle(self, bgr: np.ndarray, color: ColorResult) -> float:
        """
        Détecte un rectangle via contours + ratio d'aspect.
        Retourne un score [0-1].
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_score = 0.0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_contour_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = max(w, h) / max(min(w, h), 1)
            if self.rect_aspect_min <= aspect <= self.rect_aspect_max:
                # Plus le contour est grand et proche d'un rectangle parfait, meilleur est le score
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                if len(approx) == 4:
                    score = min(1.0, area / (bgr.shape[0] * bgr.shape[1]) * 10)
                    best_score = max(best_score, score)

        return best_score

    # ------------------------------------------------------------------
    # Utilitaire : annotation visuelle
    # ------------------------------------------------------------------

    def draw_detections(self, bgr: np.ndarray, analysis: ImageAnalysis) -> np.ndarray:
        """Dessine les détections sur une copie de l'image."""
        out = bgr.copy()
        c = analysis.color

        for bbox, color_bgr, label in [
            (c.red_bbox,        (0, 0, 255),   "Rouge"),
            (c.blue_bbox,       (255, 0, 0),   "Bleu"),
            (c.reflective_bbox, (200, 200, 200), "Reflechissant"),
        ]:
            if bbox is not None:
                x, y, w, h = bbox
                cv2.rectangle(out, (x, y), (x+w, y+h), color_bgr, 2)
                cv2.putText(out, label, (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)

        return out
