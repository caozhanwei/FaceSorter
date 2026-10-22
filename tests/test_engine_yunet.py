# -*- coding: utf-8 -*-
"""引擎A（YuNet 检测 + SFace 嵌入）：模型缺失时整文件跳过；模型在则全量跑。"""
import os, cv2, numpy as np, pytest

from app.model_paths import model_dir, has_yunet_models
from app.detector import FaceDetector, FaceBox

ASSETS = os.path.join(os.path.dirname(__file__), "assets")
pytestmark = pytest.mark.skipif(not has_yunet_models(),
                                reason="models/ 下缺 YuNet/SFace onnx（运行 tools/fetch_models.py）")


@pytest.fixture()
def engine():
    from app.engine_yunet import YuNetFaceDetector
    return YuNetFaceDetector()


def _img(name):
    return cv2.imdecode(np.fromfile(os.path.join(ASSETS, name), dtype=np.uint8),
                        cv2.IMREAD_COLOR)


@pytest.mark.skipif(not os.path.exists(os.path.join(ASSETS, "face1.jpg")),
                    reason="需要 tests/assets/face1.jpg")
def test_detect_single_face_with_row(engine):
    boxes = engine.detect(_img("face1.jpg"))
    assert len(boxes) >= 1
    b = max(boxes, key=lambda f: f.w * f.h)
    assert b.clusturable and b.w >= 60
    assert 0.0 <= b.score <= 1.0 and b.score > 0.5
    assert b.row is not None and len(b.row) == 15  # x,y,w,h + 5关键点×2 + score，embed 依赖


@pytest.mark.skipif(not os.path.exists(os.path.join(ASSETS, "noface.jpg")),
                    reason="需要 tests/assets/noface.jpg")
def test_detect_no_face_no_false_positive(engine):
    boxes = engine.detect(_img("noface.jpg"))
    assert [b for b in boxes if b.clusturable] == []


@pytest.mark.skipif(not os.path.exists(os.path.join(ASSETS, "face1.jpg")),
                    reason="需要 tests/assets/face1.jpg")
def test_embed_shape_and_normalized(engine):
    boxes = [b for b in engine.detect(_img("face1.jpg")) if b.clusturable]
    v = engine.embed(_img("face1.jpg"), boxes[0])
    assert v.shape == (128,) and v.dtype == np.float32  # SFace 输出 128 维
    assert abs(np.linalg.norm(v) - 1.0) < 1e-4


@pytest.mark.skipif(not (os.path.exists(os.path.join(ASSETS, "face1.jpg"))
                         and os.path.exists(os.path.join(ASSETS, "group.jpg"))),
                    reason="需要 tests/assets/face1.jpg 与 group.jpg")
def test_embed_same_face_more_similar_than_cross():
    # group.jpg 是 Obama+Biden 合影，face1.jpg 是 Obama 单人照：
    # SFace 应能跨照片认出同人（<0.363）并区分另一人（>0.5）
    from app.engine_yunet import YuNetFaceDetector, DEFAULT_COSINE_THRESHOLD
    eng = YuNetFaceDetector(min_cluster=40)
    img1, img2 = _img("face1.jpg"), _img("group.jpg")
    b1 = [b for b in eng.detect(img1) if b.clusturable][0]
    gboxes = [b for b in eng.detect(img2) if b.clusturable]
    assert len(gboxes) >= 2, "合影里应至少检出两张可聚类脸"
    v1 = eng.embed(img1, b1)
    dists = [1.0 - float(np.dot(v1, eng.embed(img2, g))) for g in gboxes]
    assert min(dists) < DEFAULT_COSINE_THRESHOLD  # 合影里认出同一人
    assert max(dists) > 0.5                       # 另一人明显更远
    assert 1.0 - float(np.dot(v1, eng.embed(img1, b1))) < 0.05  # 自身嵌入稳定


def test_small_face_not_clusturable(engine):
    box = FaceBox(x=0, y=0, w=30, h=40, quality=100.0, clusturable=True)
    assert not engine._gate(w=30, h=40, quality=100.0)
    assert engine._gate(w=80, h=80, quality=100.0)
    assert not engine._gate(w=80, h=80, quality=10.0)  # 模糊脸仍不许入聚类


def test_build_engine_falls_back_to_haar_when_models_missing(tmp_path, monkeypatch):
    # 打包事故防御：模型文件不在时回退 Haar，应用仍可用（旧精度）
    monkeypatch.setattr("app.model_paths.model_dir", lambda: tmp_path)
    monkeypatch.setattr("app.engine_yunet.model_dir", lambda: tmp_path)
    assert not has_yunet_models()
    from app.engine_yunet import build_detector
    det = build_detector()
    assert isinstance(det, FaceDetector)
