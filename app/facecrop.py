# -*- coding: utf-8 -*-
"""扫描期人脸裁剪：检测行(框+五点) → 对齐正方形人脸图。

两深度引擎共用（YuNet 15 维行 / SCRFD 框+关键点行同布局：x,y,w,h,kps10,score），
Haar 回退无关键点，用矩形裁剪。裁剪取自原图而非缩略图，保证人物头像清晰。
"""
import cv2, numpy as np

from app.detector import expand_box

FACE_SIZE = 160

# ArcFace 官方 112x112 五点模板（双目/鼻/双嘴角），按比例缩放即得任意尺寸的裁剪模板；
# 与嵌入对齐同一几何 → 同模板平均脸最清晰
_TEMPLATE_112 = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float64)


def umeyama_similarity(src, dst):
    """5 点 → 2x3 相似变换矩阵（旋转+均匀缩放+平移），确定性无 RANSAC。"""
    src = np.asarray(src, np.float64); dst = np.asarray(dst, np.float64)
    n = len(src)
    src_c, dst_c = src.mean(0), dst.mean(0)
    src_d, dst_d = src - src_c, dst - dst_c
    u, s, vt = np.linalg.svd(dst_d.T @ src_d / n)
    sign = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[1, 1] = -1.0
    r = u @ sign @ vt
    scale = float((s * np.diag(sign)).sum() / ((src_d ** 2).sum() / n))
    t = dst_c - scale * (r @ src_c)
    return np.hstack([scale * r, t.reshape(2, 1)]).astype(np.float32)


def template_at(size):
    return _TEMPLATE_112 * (size / 112.0)


def aligned_crop(img, row, size=FACE_SIZE):
    """按五点把脸对齐到模板并裁成 size×size；出界部分由 warpAffine 补 0。"""
    kps = np.asarray(row, np.float64).ravel()[4:14].reshape(5, 2)
    m = umeyama_similarity(kps, template_at(size))
    return cv2.warpAffine(img, m, (size, size),
                          flags=cv2.INTER_LINEAR, borderValue=0)


def rect_crop(img, box, size=FACE_SIZE, pad=0.25):
    """无关键点（Haar 回退）的矩形裁剪：扩边、补方正方形、缩放到 size。"""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = expand_box(box.x, box.y, box.w, box.h, w, h, pad=pad)
    crop = img[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else img[0:0, 0:0]
    if crop.size == 0:
        return np.zeros((size, size, 3), np.uint8)
    ch, cw = crop.shape[:2]
    side = max(ch, cw)
    sq = np.zeros((side, side, 3), np.uint8)
    sq[(side - ch) // 2:(side - ch) // 2 + ch,
       (side - cw) // 2:(side - cw) // 2 + cw] = crop
    return cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)
