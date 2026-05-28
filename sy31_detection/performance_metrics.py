"""
performance_metrics.py
-----------------------
Calcul des métriques de performance pour le système de détection :
  - Vrais Positifs (VP), Faux Positifs (FP), Faux Négatifs (FN), Vrais Négatifs (VN)
  - Précision, Rappel, F1-score
  - Matrice de confusion
  - Rapport global

Utilisation :
    metrics = PerformanceMetrics(OBJECT_LABELS)
    metrics.add(predicted="gros_carton_rouge", ground_truth="gros_carton_rouge")
    metrics.add(predicted="petit_carton_bleu",  ground_truth="gros_carton_bleu")
    metrics.report()
    metrics.plot_confusion_matrix("confusion.png")
"""

from __future__ import annotations
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ClassMetrics:
    label: str
    vp: int = 0   # Vrais Positifs
    fp: int = 0   # Faux Positifs
    fn: int = 0   # Faux Négatifs
    vn: int = 0   # Vrais Négatifs

    @property
    def precision(self) -> float:
        return self.vp / (self.vp + self.fp) if (self.vp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.vp / (self.vp + self.fn) if (self.vp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        total = self.vp + self.fp + self.fn + self.vn
        return (self.vp + self.vn) / total if total > 0 else 0.0


class PerformanceMetrics:
    """
    Accumulateur de métriques de détection/classification.

    Parameters
    ----------
    labels : liste des labels possibles (inclure "inconnu" si pertinent)
    """

    def __init__(self, labels: List[str]):
        self.labels = labels
        self._label_idx = {l: i for i, l in enumerate(labels)}
        n = len(labels)
        # Matrice de confusion : [vrai][prédit]
        self._confusion = np.zeros((n, n), dtype=int)

    # ------------------------------------------------------------------
    # Enregistrement
    # ------------------------------------------------------------------

    def add(self, predicted: str, ground_truth: str):
        """Ajoute une paire (prédiction, vérité terrain)."""
        if predicted not in self._label_idx:
            predicted = "inconnu"
        if ground_truth not in self._label_idx:
            ground_truth = "inconnu"
        i = self._label_idx[ground_truth]
        j = self._label_idx[predicted]
        self._confusion[i, j] += 1

    def add_batch(self, predictions: List[str], ground_truths: List[str]):
        for p, g in zip(predictions, ground_truths):
            self.add(p, g)

    # ------------------------------------------------------------------
    # Calcul des métriques
    # ------------------------------------------------------------------

    def per_class(self) -> Dict[str, ClassMetrics]:
        """Retourne les métriques par classe."""
        n = len(self.labels)
        result = {}
        for i, label in enumerate(self.labels):
            vp = int(self._confusion[i, i])
            fp = int(self._confusion[:, i].sum()) - vp
            fn = int(self._confusion[i, :].sum()) - vp
            vn = int(self._confusion.sum()) - vp - fp - fn
            result[label] = ClassMetrics(label=label, vp=vp, fp=fp, fn=fn, vn=vn)
        return result

    def macro_f1(self) -> float:
        """F1 macro-moyenné sur toutes les classes."""
        pc = self.per_class()
        f1s = [m.f1 for m in pc.values()]
        return float(np.mean(f1s)) if f1s else 0.0

    def global_accuracy(self) -> float:
        total = self._confusion.sum()
        correct = np.diag(self._confusion).sum()
        return float(correct / total) if total > 0 else 0.0

    # ------------------------------------------------------------------
    # Rapport textuel
    # ------------------------------------------------------------------

    def report(self) -> str:
        """Génère un rapport textuel complet."""
        lines = []
        lines.append("=" * 65)
        lines.append("RAPPORT DE PERFORMANCE – Détection d'objets SY31")
        lines.append("=" * 65)
        lines.append(f"Précision globale : {self.global_accuracy()*100:.1f}%")
        lines.append(f"F1 macro          : {self.macro_f1()*100:.1f}%")
        lines.append(f"Total échantillons: {self._confusion.sum()}")
        lines.append("")
        lines.append(f"{'Classe':<26} {'VP':>4} {'FP':>4} {'FN':>4} {'VN':>5} "
                     f"{'Préc.':>7} {'Rappel':>7} {'F1':>6}")
        lines.append("-" * 65)
        for label, m in self.per_class().items():
            if m.vp + m.fp + m.fn + m.vn == 0:
                continue
            lines.append(
                f"{label:<26} {m.vp:>4} {m.fp:>4} {m.fn:>4} {m.vn:>5} "
                f"{m.precision*100:>6.1f}% {m.recall*100:>6.1f}% {m.f1*100:>5.1f}%"
            )
        lines.append("=" * 65)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Matrice de confusion (texte et image)
    # ------------------------------------------------------------------

    def confusion_matrix_str(self) -> str:
        """Matrice de confusion en texte."""
        short = [l[:10] for l in self.labels]
        header = " " * 12 + "  ".join(f"{s:>10}" for s in short)
        lines = [header]
        for i, label in enumerate(self.labels):
            row = f"{label[:11]:<12}" + "  ".join(
                f"{self._confusion[i, j]:>10}" for j in range(len(self.labels))
            )
            lines.append(row)
        return "\n".join(lines)

    def plot_confusion_matrix(self, output_path: str):
        """Sauvegarde une image de la matrice de confusion (nécessite matplotlib)."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[WARN] matplotlib non disponible, pas de graphique généré.")
            return

        n = len(self.labels)
        fig, ax = plt.subplots(figsize=(max(8, n), max(6, n)))

        cm = self._confusion.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm / row_sums

        im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ticks = np.arange(n)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(self.labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(self.labels, fontsize=9)

        thresh = 0.5
        for i in range(n):
            for j in range(n):
                val = int(self._confusion[i, j])
                if val > 0:
                    ax.text(j, i, str(val),
                            ha="center", va="center",
                            color="white" if cm_norm[i, j] > thresh else "black",
                            fontsize=8)

        ax.set_ylabel("Vérité terrain")
        ax.set_xlabel("Prédiction")
        ax.set_title("Matrice de confusion – Détection objets SY31")
        plt.tight_layout()
        plt.savefig(output_path, dpi=120)
        plt.close(fig)
        print(f"[INFO] Matrice de confusion sauvegardée : {output_path}")
