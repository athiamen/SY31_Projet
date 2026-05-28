"""
test_classifier.py — Tests unitaires du pipeline de détection SY31.
Lance avec : python3 test/test_classifier.py
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Stubs ROS2 minimaux ──────────────────────────────────────────────────────
import types
for mod in ['rclpy','rclpy.node','rclpy.qos','rclpy.parameter_service',
            'sensor_msgs','sensor_msgs.msg','sensor_msgs_py',
            'sensor_msgs_py.point_cloud2','std_msgs','std_msgs.msg',
            'geometry_msgs','geometry_msgs.msg','visualization_msgs',
            'visualization_msgs.msg','cv_bridge','turtlebot3_msgs','turtlebot3_msgs.msg']:
    sys.modules.setdefault(mod, types.ModuleType(mod))

class _FC:
    FLOAT32=7; ADD=DELETEALL=LINE_STRIP=CYLINDER=CUBE=0
    def __init__(self,*a,**k):
        self.markers=[]; self.points=[]
        self.color=type('C',(),{'r':0,'g':0,'b':0,'a':0})()
        self.scale=type('S',(),{'x':0,'y':0,'z':0})()
        self.pose=type('P',(),{'position':type('Q',(),{'x':0,'y':0})()})()
        self.type=0

for _n in ['PointCloud2','PointField','LaserScan','Image','CompressedImage','CameraInfo']:
    setattr(sys.modules['sensor_msgs.msg'], _n, _FC)
for _n in ['String','Float32','Header']:
    setattr(sys.modules['std_msgs.msg'], _n, _FC)
sys.modules['geometry_msgs.msg'].Point = _FC
sys.modules['visualization_msgs.msg'].Marker = _FC
sys.modules['visualization_msgs.msg'].MarkerArray = _FC
sys.modules['sensor_msgs_py.point_cloud2'].read_points_numpy = lambda *a,**k: np.zeros((0,4))
sys.modules['sensor_msgs_py.point_cloud2'].create_cloud = lambda *a,**k: None

class _FP:
    def __init__(self,*a,**k): self.name=a[0] if a else ''; self.value=a[1] if len(a)>1 else 0
sys.modules['rclpy.parameter_service'].Parameter = _FP
sys.modules['rclpy.parameter_service'].SetParametersResult = type('R',(),{'__init__':lambda s,**k:None})

# ── Imports des modules testés ───────────────────────────────────────────────
from sy31_detection.lidar_analyzer    import LidarAnalyzer, LidarAnalysis, LidarCluster
from sy31_detection.detect            import detect_colors, ColorDetection
from sy31_detection.shaper_bbox       import ShaperBBox, BBoxResult
from sy31_detection.shaper_cylinder   import ShaperCylinder, CylinderResult, fit_circle
from sy31_detection.shaper_polyline   import ShaperPolyline, PolylineResult
from sy31_detection.clusterer         import Clusterer
from sy31_detection.object_classifier import ObjectClassifier, OBJECT_LABELS
from sy31_detection.performance_metrics import PerformanceMetrics


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_lidar(present=True, is_large=True, is_refl=False, dist=0.6, width=0.4):
    if not present:
        return LidarAnalysis(object_present=False)
    c = LidarCluster(
        angle_start_deg=-5, angle_end_deg=5,
        distance_m=dist, distance_min_m=dist-0.05,
        width_angular_deg=10, width_metric_m=width,
        mean_intensity=0.8 if is_refl else 0.1,
        is_reflective=is_refl,
    )
    return LidarAnalysis(clusters=[c], front_cluster=c, object_present=True)

def make_color(red=False, blue=False, refl=False):
    return ColorDetection(
        has_red=red, has_blue=blue, has_reflective=refl,
        red_ratio=0.15 if red else 0.0,
        blue_ratio=0.15 if blue else 0.0,
        reflective_ratio=0.10 if refl else 0.0,
    )

def make_circle_bbox(r=0.15):
    return BBoxResult(cx=0.5, cy=0, width=2*r, length=2*r,
                      aspect_ratio=1.0, area=(2*r)**2)

def make_rect_bbox(w=0.4, l=0.15):
    return BBoxResult(cx=0.5, cy=0, width=w, length=l,
                      aspect_ratio=w/l, area=w*l)

def make_cylinder(residual=0.02):
    return CylinderResult(cx=0.5, cy=0, radius=0.15, residual=residual)

def make_polyline(angle=80.0):
    return PolylineResult(points=[[0,0],[0.2,0],[0.2,0.3]], n_segments=2, max_angle_deg=angle)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestTransformer:
    def test_conversion(self):
        from sy31_detection.transformer import Transformer
        from types import SimpleNamespace
        msg = SimpleNamespace(
            angle_min=0.0, angle_max=np.radians(90),
            angle_increment=np.radians(1.0),
            range_min=0.12, range_max=3.5,
            ranges=[1.0]*90, intensities=[50.0]*90,
        )
        x, y, i = Transformer.scan_to_cartesian(msg)
        assert len(x) == 90
        assert abs(x[0] - 1.0) < 0.01   # theta=0 → x=r, y=0
        assert abs(y[0]) < 0.01
        print("  ✓ test_conversion")

    def test_filters_invalid(self):
        from sy31_detection.transformer import Transformer
        from types import SimpleNamespace
        msg = SimpleNamespace(
            angle_min=0.0, angle_max=np.radians(5),
            angle_increment=np.radians(1.0),
            range_min=0.12, range_max=3.5,
            ranges=[0.0, 0.5, float('inf'), 4.0, 1.0], intensities=[],
        )
        x, _, _ = Transformer.scan_to_cartesian(msg)
        assert len(x) == 2   # seuls 0.5 et 1.0 passent
        print("  ✓ test_filters_invalid")


class TestIntensityFilter:
    def test_filter(self):
        from sy31_detection.intensity_filter import IntensityFilter
        pts = np.array([[0,0,10],[1,1,200],[2,2,50]], dtype=float)
        out = IntensityFilter.filter(pts, threshold=100)
        assert len(out) == 1 and out[0, 2] == 200
        print("  ✓ test_filter")


class TestClusterer:
    def test_two_clusters(self):
        xy = np.array([[0,0],[0.1,0],[0.2,0],[5,5],[5.1,5]], dtype=float)
        ids = Clusterer.dbscan(xy, eps=0.2, min_pts=2)
        assert len(set(ids[ids >= 0])) == 2
        print(f"  ✓ test_two_clusters → ids={ids}")

    def test_noise(self):
        xy = np.array([[0,0],[10,10]], dtype=float)
        ids = Clusterer.dbscan(xy, eps=0.1, min_pts=2)
        assert all(ids == -1)
        print("  ✓ test_noise")


class TestShapers:
    def _pts_with_id(self, xy, cid=0):
        c = np.full((len(xy), 1), cid)
        return np.hstack([xy, np.zeros((len(xy),1)), c])

    def test_bbox(self):
        xy = np.array([[0,0],[1,0],[1,0.5],[0,0.5]], dtype=float)
        pts = self._pts_with_id(xy)
        res = ShaperBBox.fit(pts)
        assert len(res) == 1
        r = res[0]
        assert abs(r.width - 1.0) < 0.01
        assert abs(r.length - 0.5) < 0.01
        assert abs(r.aspect_ratio - 2.0) < 0.01
        print(f"  ✓ test_bbox  w={r.width:.2f} l={r.length:.2f} ar={r.aspect_ratio:.2f}")

    def test_fit_circle(self):
        theta = np.linspace(0, 2*np.pi, 30)
        xy = np.column_stack([1.5 + 0.4*np.cos(theta), 0.3 + 0.4*np.sin(theta)])
        r = fit_circle(xy)
        assert abs(r.cx - 1.5) < 0.01
        assert abs(r.cy - 0.3) < 0.01
        assert abs(r.radius - 0.4) < 0.01
        assert r.residual < 0.01   # cercle parfait → faible résidu
        print(f"  ✓ test_fit_circle  residual={r.residual:.4f}")

    def test_rdp_straight_line(self):
        line = np.array([[0,0],[0.5,0.005],[1,0],[1.5,0.005],[2,0]], dtype=float)
        s = ShaperPolyline.rdp(line, eps=0.05)
        assert len(s) == 2   # quasi-droite → 2 points
        print(f"  ✓ test_rdp  {len(line)} pts → {len(s)}")

    def test_polyline_angle(self):
        # Angle droit → max_angle_deg ≈ 90
        pts = np.array([[0,0],[1,0],[1,1]], dtype=float)
        ppts = self._pts_with_id(pts)
        res = ShaperPolyline.fit(ppts, eps=0.001)
        assert res[0].max_angle_deg > 80
        print(f"  ✓ test_polyline_angle  max={res[0].max_angle_deg:.1f}°")


class TestObjectClassifier:
    def _clf(self):
        return ObjectClassifier(confirmation_frames=1, min_score=0.20)

    def test_panneau_rond(self):
        clf = self._clf()
        det = clf.classify(
            lidar    = make_lidar(is_large=False, width=0.25),
            color    = make_color(red=True),
            cylinder = make_cylinder(residual=0.01),  # très circulaire
            bbox     = make_circle_bbox(),
        )
        assert det is not None
        assert det.shape == "circle"
        assert "panneau" in det.label or "rouge" in det.label
        print(f"  ✓ test_panneau_rond → {det.label} ({det.score:.2f})")

    def test_panneau_rectangulaire(self):
        clf = self._clf()
        det = clf.classify(
            lidar    = make_lidar(is_large=False),
            color    = make_color(),
            polyline = make_polyline(angle=85.0),   # coin à 85°
            bbox     = make_rect_bbox(w=0.4, l=0.15),
        )
        assert det is not None
        assert det.shape == "rectangle"
        assert "panneau" in det.label or "rectangulaire" in det.label
        print(f"  ✓ test_panneau_rect → {det.label} ({det.score:.2f})")

    def test_gros_carton_rouge(self):
        clf = self._clf()
        det = clf.classify(make_lidar(is_large=True), make_color(red=True))
        assert det is not None and "rouge" in det.label
        print(f"  ✓ test_gros_carton_rouge → {det.label}")

    def test_gros_carton_bleu(self):
        clf = self._clf()
        det = clf.classify(make_lidar(is_large=True), make_color(blue=True))
        assert det is not None and "bleu" in det.label
        print(f"  ✓ test_gros_carton_bleu → {det.label}")

    def test_petit_carton_rouge(self):
        clf = self._clf()
        det = clf.classify(make_lidar(is_large=False, width=0.12), make_color(red=True))
        assert det is not None and "rouge" in det.label
        print(f"  ✓ test_petit_carton_rouge → {det.label}")

    def test_reflective_large(self):
        clf = self._clf()
        det = clf.classify(make_lidar(is_large=True, is_refl=True), make_color(refl=True))
        assert det is not None and ("refl" in det.label or "vitre" in det.label)
        print(f"  ✓ test_reflective_large → {det.label}")

    def test_no_detection(self):
        clf = self._clf()
        det = clf.classify(LidarAnalysis(object_present=False), make_color())
        assert det is None
        print("  ✓ test_no_detection → None")

    def test_confirmation(self):
        clf = ObjectClassifier(confirmation_frames=3, min_score=0.20)
        lidar = make_lidar(is_large=True)
        color = make_color(red=True)
        d1 = clf.classify(lidar, color)
        d2 = clf.classify(lidar, color)
        d3 = clf.classify(lidar, color)
        assert d3 is not None and d3.confirmed
        print(f"  ✓ test_confirmation → confirmé à frame 3 ({d3.label})")


class TestPerformanceMetrics:
    def test_perfect(self):
        m = PerformanceMetrics(["a","b"])
        m.add("a","a"); m.add("b","b")
        assert m.global_accuracy() == 1.0
        print("  ✓ test_perfect")

    def test_fp_fn(self):
        m = PerformanceMetrics(["a","b"])
        m.add("a","a"); m.add("b","a"); m.add("b","b"); m.add("a","b")
        pc = m.per_class()
        assert pc["a"].fp == 1 and pc["a"].fn == 1
        print("  ✓ test_fp_fn")


# ── Runner ────────────────────────────────────────────────────────────────────

def run_all():
    suites = [TestTransformer, TestIntensityFilter, TestClusterer,
              TestShapers, TestObjectClassifier, TestPerformanceMetrics]
    total = failed = 0
    for cls in suites:
        print(f"\n── {cls.__name__} ──")
        obj = cls()
        for name in [m for m in dir(obj) if m.startswith("test_")]:
            total += 1
            try:
                getattr(obj, name)()
            except Exception as e:
                print(f"  ✗ {name} : {e}")
                failed += 1
    print(f"\n{'='*45}")
    print(f"Tests : {total-failed}/{total} réussis")
    if failed:
        print(f"⚠  {failed} échec(s)"); sys.exit(1)
    else:
        print("✓ Tous les tests passent")

if __name__ == "__main__":
    run_all()
