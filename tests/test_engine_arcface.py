# -*- coding: utf-8 -*-
"""引擎B（SCRFD det_10g 检测 + ArcFace w600k_r50 嵌入，onnxruntime CPU）。"""
import os, cv2, numpy as np, pytest

from app.model_paths import has_arcface_models
from app.detector import FaceDetector

ASSETS = os.path.join(os.path.dirname(__file__), "assets")
pytestmark = pytest.mark.skipif(not has_arcface_models(),
                                reason="models/ 下缺 det_10g/w600k_r50（运行 tools/fetch_models.py）")


@pytest.fixture(scope="module")
def engine():
    from app.engine_arcface import ArcFaceDetector
    return ArcFaceDetector()


def _img(name):
    return cv2.imdecode(np.fromfile(os.path.join(ASSETS, name), dtype=np.uint8),
                        cv2.IMREAD_COLOR)


@pytest.mark.skipif(not os.path.exists(os.path.join(ASSETS, "face1.jpg")),
                    reason="需要 tests/assets/face1.jpg")
def test_detect_single_face_with_kps(engine):
    boxes = engine.detect(_img("face1.jpg"))
    assert len(boxes) >= 1
    b = max(boxes, key=lambda f: f.w * f.h)
    assert b.clusturable and b.w >= 60
    assert b.score > 0.5
    assert b.row is not None and len(b.row) == 15  # 框4+关键点10+score，embed 对齐依赖
    # 5 个关键点应落在框内（对齐正确性的必要条件）
    kps = b.row[4:14].reshape(5, 2)
    assert (kps[:, 0] >= b.x).all() and (kps[:, 0] <= b.x + b.w).all()
    assert (kps[:, 1] >= b.y).all() and (kps[:, 1] <= b.y + b.h).all()


@pytest.mark.skipif(not os.path.exists(os.path.join(ASSETS, "noface.jpg")),
                    reason="需要 tests/assets/noface.jpg")
def test_detect_no_face(engine):
    assert [b for b in engine.detect(_img("noface.jpg")) if b.clusturable] == []


@pytest.mark.skipif(not (os.path.exists(os.path.join(ASSETS, "face1.jpg"))
                         and os.path.exists(os.path.join(ASSETS, "group.jpg"))),
                    reason="需要 tests/assets/face1.jpg 与 group.jpg")
def test_embed_shape_and_cross_person():
    # 合影里的脸较小（<60px），单独放宽聚类门槛只考察嵌入质量
    from app.engine_arcface import ArcFaceDetector
    eng = ArcFaceDetector(min_cluster=40)
    img1, img2 = _img("face1.jpg"), _img("group.jpg")
    boxes = [b for b in eng.detect(img1) if b.clusturable]
    v1 = eng.embed(img1, boxes[0])
    assert v1.shape == (512,) and v1.dtype == np.float32  # ArcFace R50 是 512 维
    assert abs(np.linalg.norm(v1) - 1.0) < 1e-4
    # Obama 单人照 vs Obama+Biden 合影：近的一个是同人，远的是另一人
    gboxes = [b for b in eng.detect(img2) if b.clusturable]
    assert len(gboxes) >= 2
    dists = [1.0 - float(np.dot(v1, eng.embed(img2, g))) for g in gboxes]
    assert min(dists) < 0.40  # 跨照片认出同一人
    assert max(dists) > 0.55  # 另一人明显更远


def test_build_fallback_when_models_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("app.engine_arcface.model_dir", lambda: tmp_path)
    from app.engine_arcface import build_detector
    assert isinstance(build_detector(), FaceDetector)


def test_intra_threads_option_recorded():
    # ORT 线程上限可注入（多线程扫描防超订）；None=ORT 默认
    from app.engine_arcface import ArcFaceDetector
    det = ArcFaceDetector(intra_threads=3)
    assert det.intra_threads == 3
