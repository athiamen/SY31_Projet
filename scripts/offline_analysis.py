#!/usr/bin/env python3
"""
offline_analysis.py
-------------------
Analyse hors-ligne d'un fichier bag ROS2 (.mcap) pour la détection d'objets.

Permet de tester et d'évaluer le système sans lancer ROS2.
Génère un rapport de performance et des images annotées.

Usage :
    python3 scripts/offline_analysis.py --bag objets_0.mcap
    python3 scripts/offline_analysis.py --bag objets_0.mcap --output results/ --visualize
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np

# Ajout du dossier parent au path pour les imports relatifs
sys.path.insert(0, str(Path(__file__).parent.parent))

from sy31_detection.lidar_analyzer   import LidarAnalyzer, LidarAnalysis
from sy31_detection.color_detector   import ColorDetector, ImageAnalysis
from sy31_detection.object_classifier import ObjectClassifier, OBJECT_LABELS
from sy31_detection.performance_metrics import PerformanceMetrics


def parse_args():
    p = argparse.ArgumentParser(description="Analyse hors-ligne détection objets SY31")
    p.add_argument("--bag",        required=True,            help="Fichier .mcap")
    p.add_argument("--output",     default="results",        help="Dossier de sortie")
    p.add_argument("--visualize",  action="store_true",      help="Sauvegarder les frames annotées")
    p.add_argument("--max-frames", type=int, default=0,      help="Limiter le nombre de frames (0 = tout)")
    p.add_argument("--topic-img",  default="/turtlecam/image_raw/compressed")
    p.add_argument("--topic-scan", default="/scan")
    p.add_argument("--topic-sonar",default="/sensor_state")
    p.add_argument("--gt-file",    default=None,
                   help="Fichier CSV vérité terrain : timestamp_ns,label")
    return p.parse_args()


def load_ground_truth(path: str) -> dict:
    """Charge un fichier CSV de vérité terrain : {timestamp_ns: label}."""
    gt = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt[int(row["timestamp_ns"])] = row["label"]
    return gt


def find_nearest_gt(gt: dict, ts: int, tolerance_ns: int = 100_000_000) -> Optional[str]:
    """Retourne le label GT le plus proche du timestamp donné."""
    if not gt:
        return None
    best_ts = min(gt.keys(), key=lambda t: abs(t - ts))
    if abs(best_ts - ts) <= tolerance_ns:
        return gt[best_ts]
    return None


class SyncBuffer:
    """
    Maintient le dernier message LiDAR et ultrason pour synchronisation
    avec les frames image.
    """
    def __init__(self):
        self.lidar: Optional[LidarAnalysis] = None
        self.sonar: float = 0.0
        self.lidar_ts: int = 0
        self.sonar_ts: int = 0


def main():
    args = parse_args()

    # ── Imports mcap ────────────────────────────────────────────────────
    try:
        from mcap_ros2.reader import read_ros2_messages
    except ImportError:
        print("[ERREUR] Installez mcap-ros2-support : pip install mcap mcap-ros2-support")
        sys.exit(1)

    # ── Création dossier sortie ──────────────────────────────────────────
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    if args.visualize:
        frames_dir.mkdir(exist_ok=True)

    # ── Composants ──────────────────────────────────────────────────────
    lidar_analyzer = LidarAnalyzer(
        front_angle_deg = 30.0,
        max_dist_m      = 2.5,
        cluster_gap_m   = 0.15,
    )
    color_detector = ColorDetector()
    classifier     = ObjectClassifier(
        confirmation_frames = 3,
        min_score           = 0.35,
    )
    metrics = PerformanceMetrics(OBJECT_LABELS + ["none"])

    # ── Vérité terrain ──────────────────────────────────────────────────
    gt_data = {}
    if args.gt_file and Path(args.gt_file).exists():
        gt_data = load_ground_truth(args.gt_file)
        print(f"[INFO] Vérité terrain chargée : {len(gt_data)} entrées")

    # ── Lecture du bag multi-topic ───────────────────────────────────────
    buf = SyncBuffer()
    detections_log: List[dict] = []

    topics = [args.topic_img, args.topic_scan, args.topic_sonar]

    frame_count = 0
    detection_count = 0
    t_start = time.time()

    print(f"[INFO] Lecture du bag : {args.bag}")

    # Pré-lecture des topics LiDAR et ultrason pour synchronisation
    lidar_cache: dict = {}
    sonar_cache: dict = {}

    print("[INFO] Chargement LiDAR et ultrason...")
    for msg in read_ros2_messages(args.bag, topics=[args.topic_scan]):
        ts = msg.log_time
        m  = msg.ros_msg
        lidar_cache[ts] = m

    for msg in read_ros2_messages(args.bag, topics=[args.topic_sonar]):
        ts = msg.log_time
        m  = msg.ros_msg
        sonar_cache[ts] = float(m.sonar)

    lidar_timestamps = sorted(lidar_cache.keys())
    sonar_timestamps = sorted(sonar_cache.keys())
    print(f"[INFO] {len(lidar_timestamps)} scans LiDAR, {len(sonar_timestamps)} mesures ultrason")

    # ── Traitement frame par frame ───────────────────────────────────────
    print("[INFO] Traitement des frames image...")
    for msg in read_ros2_messages(args.bag, topics=[args.topic_img]):
        if args.max_frames > 0 and frame_count >= args.max_frames:
            break

        ts = msg.log_time
        m  = msg.ros_msg

        # Décodage image
        data = bytes(m.data)
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue

        # Synchronisation LiDAR (scan le plus proche en temps)
        # log_time est un datetime.datetime
        from datetime import timedelta
        MAX_DELTA = timedelta(milliseconds=200)

        nearest_lidar_m = None
        if lidar_timestamps:
            nearest_ts = min(lidar_timestamps, key=lambda t: abs(t - ts))
            if abs(nearest_ts - ts) < MAX_DELTA:
                lm = lidar_cache[nearest_ts]
                nearest_lidar_m = lidar_analyzer.analyze(
                    ranges          = lm.ranges,
                    intensities     = lm.intensities if len(lm.intensities) > 0 else None,
                    angle_min       = lm.angle_min,
                    angle_increment = lm.angle_increment,
                )

        if nearest_lidar_m is None:
            from sy31_detection.lidar_analyzer import LidarAnalysis
            nearest_lidar_m = LidarAnalysis()

        # Synchronisation ultrason
        sonar_val = 0.0
        if sonar_timestamps:
            nearest_sonar_ts = min(sonar_timestamps, key=lambda t: abs(t - ts))
            if abs(nearest_sonar_ts - ts) < MAX_DELTA:
                sonar_val = sonar_cache[nearest_sonar_ts]

        # Analyse couleur
        image_analysis = color_detector.analyze(img)

        # Classification
        det = classifier.classify(
            lidar     = nearest_lidar_m,
            color     = image_analysis.color,
            sonar_raw = sonar_val,
        )

        label_pred = det.label if det is not None else "none"
        score      = det.score if det is not None else 0.0
        confirmed  = det.confirmed if det is not None else False

        if det is not None:
            detection_count += 1

        # Vérité terrain (optionnel)
        gt_label = find_nearest_gt(gt_data, ts) if gt_data else None
        if gt_label:
            metrics.add(predicted=label_pred, ground_truth=gt_label)

        # Log
        log_entry = {
            "frame":     frame_count,
            "timestamp": ts,
            "label":     label_pred,
            "score":     f"{score:.3f}",
            "confirmed": confirmed,
            "gt":        gt_label or "",
            "lidar_dist":  f"{det.lidar_dist_m:.3f}" if det else "-1",
            "lidar_width": f"{det.lidar_width_m:.3f}" if det else "-1",
            "color":       det.color if det else "none",
            "shape":       det.shape if det else "none",
        }
        detections_log.append(log_entry)

        # Affichage progression
        if frame_count % 100 == 0:
            elapsed = time.time() - t_start
            print(f"  Frame {frame_count:4d} | détections={detection_count} | {elapsed:.1f}s | "
                  f"last={label_pred}")

        # Sauvegarde frame annotée
        if args.visualize and (det is not None and det.confirmed):
            annotated = color_detector.draw_detections(img, image_analysis)
            text = f"{det.label} ({det.score*100:.0f}%)"
            cv2.putText(annotated, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            if det.lidar_dist_m > 0:
                cv2.putText(annotated, f"dist={det.lidar_dist_m:.2f}m", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            frame_path = frames_dir / f"frame_{frame_count:05d}_{det.label}.jpg"
            cv2.imwrite(str(frame_path), annotated)

        frame_count += 1

    elapsed = time.time() - t_start
    print(f"\n[INFO] Traitement terminé : {frame_count} frames en {elapsed:.1f}s "
          f"({frame_count/elapsed:.1f} fps)")
    print(f"[INFO] Détections : {detection_count} ({detection_count/max(frame_count,1)*100:.1f}%)")

    # ── Sauvegarde des résultats ─────────────────────────────────────────
    # CSV des détections
    csv_path = out_dir / "detections.csv"
    if detections_log:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=detections_log[0].keys())
            writer.writeheader()
            writer.writerows(detections_log)
        print(f"[INFO] Détections sauvegardées : {csv_path}")

    # Statistiques par label
    label_counts: dict = {}
    for entry in detections_log:
        lbl = entry["label"]
        if lbl != "none":
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
    print("\n[INFO] Détections par objet :")
    for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {lbl:<26} : {cnt}")

    # Rapport de performance (si GT disponible)
    if gt_data:
        report = metrics.report()
        print("\n" + report)
        report_path = out_dir / "performance.txt"
        report_path.write_text(report)
        print(f"[INFO] Rapport sauvegardé : {report_path}")

        cm_path = out_dir / "confusion_matrix.png"
        metrics.plot_confusion_matrix(str(cm_path))
    else:
        print("\n[INFO] Pas de fichier de vérité terrain fourni.")
        print("       Pour évaluer les performances, créez un fichier CSV avec :")
        print("       timestamp_ns,label")
        print("       et passez-le via --gt-file")

    print(f"\n[OK] Résultats dans : {out_dir.resolve()}")


if __name__ == "__main__":
    main()
