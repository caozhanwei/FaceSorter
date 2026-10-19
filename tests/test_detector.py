import os, cv2, numpy as np, pytest
from app.detector import FaceDetector, FaceBox, expand_box, nms

ASSETS = os.path.join(os.path.dirname(__file__), "assets")

def test_expand_box_clamps():
    assert expand_box(0, 0, 10, 10, 100, 100) == (0, 0, 12, 12)
    assert expand_box(95, 95, 10, 10, 100, 100) == (93, 93, 100, 100)

def test_nms_suppresses_overlap():
    boxes = [(0, 0, 100, 100), (5, 5, 100, 100), (200, 200, 50, 50)]
    kept = nms(boxes, 0.3)
    assert (0, 0, 100, 100) in kept and (200, 200, 50, 50) in kept
    assert (5, 5, 100, 100) not in kept

def test_quality_gate_small_face_not_clusturable():
    det = FaceDetector()
    fb = FaceBox(x=0, y=0, w=40, h=40, quality=100.0, clusturable=False)
    assert not fb.clusturable
    assert not det._gate(w=40, h=40, quality=100.0)
    assert det._gate(w=80, h=80, quality=100.0)
    assert not det._gate(w=80, h=80, quality=10.0)

@pytest.mark.skipif(not os.path.exists(os.path.join(ASSETS, "face1.jpg")),
                    reason="需要 tests/assets/face1.jpg")
def test_detect_real_single_face():
    img = cv2.imdecode(np.fromfile(os.path.join(ASSETS, "face1.jpg"), dtype=np.uint8),
                       cv2.IMREAD_COLOR)
    boxes = FaceDetector().detect(img)
    assert len(boxes) >= 1
    b = max(boxes, key=lambda f: f.w * f.h)
    assert b.clusturable and b.w >= 60

@pytest.mark.skipif(not os.path.exists(os.path.join(ASSETS, "noface.jpg")),
                    reason="需要 tests/assets/noface.jpg")
def test_detect_real_no_face_low_false_positive():
    img = cv2.imdecode(np.fromfile(os.path.join(ASSETS, "noface.jpg"), dtype=np.uint8),
                       cv2.IMREAD_COLOR)
    boxes = FaceDetector().detect(img)
    assert len([b for b in boxes if b.clusturable]) == 0
