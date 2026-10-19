# -*- coding: utf-8 -*-
"""扫描期人脸裁剪：统一对齐裁剪（两引擎共用）与 Haar 回退矩形裁剪。"""
import numpy as np, cv2, pytest
from app.facecrop import umeyama_similarity, aligned_crop, rect_crop, FACE_SIZE


def _kps():
    # 五点须与模板成相似形（真实脸即如此），umeyama 最小二乘拟合才能精确落位
    from app.facecrop import template_at
    return template_at(220).copy()


def _row(kps):
    return np.concatenate([[50, 55, 100, 110], kps.ravel(), [0.9]]).astype(np.float32)


def _dot_img(kps, size=220):
    img = np.zeros((size, size, 3), np.uint8)
    for i, (x, y) in enumerate(kps):
        color = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0)][i]
        cv2.circle(img, (int(x), int(y)), 4, color, -1)
    return img


def test_umeyama_maps_points_exactly():
    kps = _kps()
    # 已知刚体平移：所有点 +30,+10 后应被矩阵精确映射
    m = umeyama_similarity(kps, kps + np.array([30.0, 10.0]))
    proj = (m[:, :2] @ kps.T).T + m[:, 2]
    assert np.allclose(proj, kps + np.array([30.0, 10.0]), atol=1e-6)


def test_aligned_crop_maps_landmarks_to_template():
    img = _dot_img(_kps())
    out = aligned_crop(img, _row(_kps()), size=FACE_SIZE)
    assert out.shape == (FACE_SIZE, FACE_SIZE, 3)
    # 模板五点（缩放后）位置应是对应颜色圆点
    from app.facecrop import template_at
    tmpl = template_at(FACE_SIZE)
    checks = [((46, 62), (0, 0, 255)), ((114, 62), (0, 255, 0)),
              ((80, 96), (255, 0, 0)), ((50, 124), (0, 255, 255)),
              ((110, 124), (255, 255, 0))]
    for (tx, ty), expect in zip(tmpl, [c[1] for c in checks]):
        px = out[int(ty), int(tx)]
        # 插值容忍：目标通道显著高于其余通道
        assert px.tolist() == pytest.approx(list(expect), abs=40), (px, expect, (tx, ty))


def test_aligned_crop_out_of_bounds_survives():
    # 五点贴边/出界的脸也要能出图（warpAffine 边界填 0，不崩）
    kps = np.array([[1, 1], [30, 1], [15, 20], [5, 30], [25, 30]], dtype=np.float64)
    img = _dot_img(kps, size=64)
    out = aligned_crop(img, _row(kps) * 0 + np.concatenate(
        [[0, 0, 32, 32], kps.ravel(), [0.9]]).astype(np.float32))
    assert out.shape == (FACE_SIZE, FACE_SIZE, 3)


def test_rect_crop_square_and_size():
    from app.detector import FaceBox
    img = np.zeros((400, 600, 3), np.uint8)
    box = FaceBox(100, 50, 120, 160, 100.0, True)  # 竖长脸
    out = rect_crop(img, box, size=FACE_SIZE)
    assert out.shape == (FACE_SIZE, FACE_SIZE, 3)
