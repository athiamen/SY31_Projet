"""
lidar_analyzer.py
-----------------
Analyse un scan LiDAR (sensor_msgs/LaserScan) pour extraire :
  - distance à l'objet frontal
  - largeur angulaire et largeur métrique estimée
  - présence d'intensités réfléchissantes
  - liste de clusters (objets distincts dans le champ avant)
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class LidarCluster:
    """Un groupe de rayons LiDAR correspondant à un objet potentiel."""
    angle_start_deg: float
    angle_end_deg:   float
    distance_m:      float          # Distance moyenne du cluster
    distance_min_m:  float          # Distance minimale (point le plus proche)
    width_angular_deg: float        # Largeur angulaire
    width_metric_m:  float          # Largeur métrique estimée (arc)
    mean_intensity:  float          # Intensité moyenne (réflectivité)
    is_reflective:   bool           # True si intensité élevée

    @property
    def is_large(self) -> bool:
        """Heuristique taille : large si > 30 cm à distance nominale."""
        return self.width_metric_m > 0.30

    @property
    def center_angle_deg(self) -> float:
        return (self.angle_start_deg + self.angle_end_deg) / 2.0


@dataclass
class LidarAnalysis:
    """Résultat complet d'une analyse de scan."""
    clusters: List[LidarCluster] = field(default_factory=list)
    front_cluster: Optional[LidarCluster] = None   # cluster le plus centré
    object_present: bool = False
    raw_ranges: np.ndarray = field(default_factory=lambda: np.array([]))


class LidarAnalyzer:
    """
    Analyse un scan LiDAR pour détecter et caractériser des objets.

    Parameters
    ----------
    front_angle_deg : demi-angle du secteur frontal (±front_angle_deg)
    max_dist_m      : distance max pour considérer un objet
    cluster_gap_m   : saut de distance entre deux rayons pour séparer les clusters
    refl_threshold  : seuil d'intensité normalisée pour déclarer un objet réfléchissant
    range_max_valid : valeur max considérée valide (filtre inf/nan/range_max)
    """

    def __init__(
        self,
        front_angle_deg: float = 30.0,
        max_dist_m: float = 2.5,
        cluster_gap_m: float = 0.15,
        refl_threshold: float = 0.6,
        range_max_valid: float = 3.4,
    ):
        self.front_angle_deg = front_angle_deg
        self.max_dist_m = max_dist_m
        self.cluster_gap_m = cluster_gap_m
        self.refl_threshold = refl_threshold
        self.range_max_valid = range_max_valid

    # ------------------------------------------------------------------
    # Interface principale
    # ------------------------------------------------------------------

    def analyze(self, ranges, intensities=None,
                angle_min: float = 0.0,
                angle_increment: float = math.radians(1.0)) -> LidarAnalysis:
        """
        Analyse un scan complet.

        Parameters
        ----------
        ranges          : tableau de distances (float, en mètres)
        intensities     : tableau d'intensités (float, optionnel)
        angle_min       : angle du premier rayon (radians)
        angle_increment : incrément angulaire entre rayons (radians)
        """
        ranges = np.asarray(ranges, dtype=float)
        n = len(ranges)

        # Normalisation intensités
        if intensities is not None and len(intensities) == n:
            intens = np.asarray(intensities, dtype=float)
            max_i = intens.max()
            if max_i > 0:
                intens = intens / max_i
        else:
            intens = np.zeros(n)

        # Nettoyage : remplace 0, inf, nan et > range_max par np.inf
        clean = ranges.copy()
        clean[(clean <= 0) | (clean > self.range_max_valid) | ~np.isfinite(clean)] = np.inf

        # Calcul des angles de chaque rayon (degrés)
        angles_deg = np.degrees(angle_min + np.arange(n) * angle_increment)

        # Secteur frontal : angles proches de 0° (ou 360°)
        # On considère 0° = devant, la plage est ]-front, +front]
        fa = self.front_angle_deg
        front_mask = (
            (angles_deg <= fa) |
            (angles_deg >= (360.0 - fa))
        )

        # Sélection rayons frontaux valides et proches
        valid_front = front_mask & (clean < self.max_dist_m)

        if not valid_front.any():
            return LidarAnalysis(object_present=False, raw_ranges=clean)

        # Extraction des clusters dans la zone frontale
        clusters = self._extract_clusters(
            clean, intens, angles_deg, valid_front
        )

        # Cluster le plus centré (angle le plus proche de 0° ou 360°)
        front_cluster = self._find_front_cluster(clusters)

        return LidarAnalysis(
            clusters=clusters,
            front_cluster=front_cluster,
            object_present=len(clusters) > 0,
            raw_ranges=clean,
        )

    # ------------------------------------------------------------------
    # Méthodes internes
    # ------------------------------------------------------------------

    def _extract_clusters(
        self,
        ranges: np.ndarray,
        intens: np.ndarray,
        angles_deg: np.ndarray,
        mask: np.ndarray,
    ) -> List[LidarCluster]:
        """Regroupe les rayons valides consécutifs en clusters."""
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return []

        clusters: List[LidarCluster] = []
        current = [indices[0]]

        for idx in indices[1:]:
            prev = current[-1]
            # Séparation si distance entre rayons trop grande OU si les indices
            # ne sont pas contigus (saut d'angle dans le scan)
            gap_dist = abs(float(ranges[idx]) - float(ranges[prev]))
            angle_gap = abs(float(angles_deg[idx]) - float(angles_deg[prev]))
            # gestion du wrap autour de 0/360
            if angle_gap > 180:
                angle_gap = 360 - angle_gap

            if gap_dist > self.cluster_gap_m or angle_gap > 3.0:
                c = self._make_cluster(current, ranges, intens, angles_deg)
                if c is not None:
                    clusters.append(c)
                current = [idx]
            else:
                current.append(idx)

        c = self._make_cluster(current, ranges, intens, angles_deg)
        if c is not None:
            clusters.append(c)

        return clusters

    def _make_cluster(
        self,
        indices: List[int],
        ranges: np.ndarray,
        intens: np.ndarray,
        angles_deg: np.ndarray,
    ) -> Optional[LidarCluster]:
        if not indices:
            return None

        r = ranges[indices]
        valid = r < np.inf
        if not valid.any():
            return None

        r_valid = r[valid]
        dist_mean = float(r_valid.mean())
        dist_min = float(r_valid.min())
        ang_start = float(angles_deg[indices[0]])
        ang_end = float(angles_deg[indices[-1]])
        ang_width = float(abs(ang_end - ang_start))
        if ang_width > 180:
            ang_width = 360 - ang_width

        # Largeur métrique estimée par l'arc à la distance moyenne
        width_m = dist_mean * math.radians(ang_width)

        mean_i = float(intens[indices].mean())
        is_refl = mean_i >= self.refl_threshold

        return LidarCluster(
            angle_start_deg=ang_start,
            angle_end_deg=ang_end,
            distance_m=dist_mean,
            distance_min_m=dist_min,
            width_angular_deg=ang_width,
            width_metric_m=width_m,
            mean_intensity=mean_i,
            is_reflective=is_refl,
        )

    def _find_front_cluster(self, clusters: List[LidarCluster]) -> Optional[LidarCluster]:
        """Retourne le cluster dont le centre est le plus proche de 0°."""
        if not clusters:
            return None

        def angular_dist_to_zero(c: LidarCluster) -> float:
            a = c.center_angle_deg % 360
            return min(a, 360 - a)

        return min(clusters, key=angular_dist_to_zero)
